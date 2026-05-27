"""Video loading via NVIDIA PyNvVideoCodec (direct NVDEC), as an alternative to DALI.

This module provides a drop-in replacement for the DALI-based prediction loader,
producing the same `UnlabeledBatchDict` outputs that `predict_step` consumes.
It is designed for inference only (`train_stage="predict"`).

Single-video, single-view, base (non-context) models are supported. Context (5-frame)
and multi-view modes raise `NotImplementedError` and should fall back to DALI.

Implementation notes:
    * Uses `PyNvVideoCodec.ThreadedDecoder`, which prefetches frames in a background
      thread so NVDEC work overlaps with the model forward pass.
    * Frames are requested as `OutputColorType.RGBP` (planar RGB, CHW) so no
      channel-permute is needed on the consumer side.
    * Frames are exchanged via DLPack (`torch.from_dlpack`) for zero-copy GPU-to-GPU
      transfer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.datatypes import UnlabeledBatchDict

__all__ = [
    'LitNvCodecWrapper',
    'PrepareNvCodec',
]


def _import_pynvc():
    """Import PyNvVideoCodec lazily with a helpful error message."""
    try:
        import PyNvVideoCodec as nvc  # type: ignore
    except ImportError as e:
        raise ImportError(
            'PyNvVideoCodec is required for the nvcodec video backend. '
            'Install with: pip install PyNvVideoCodec'
        ) from e
    return nvc


class LitNvCodecWrapper:
    """Iterator that decodes a single video on NVDEC and yields `UnlabeledBatchDict` batches.

    Mimics the interface of `lightning_pose.data.dali.LitDaliWrapper` so it can be
    passed directly to `pl.Trainer.predict(..., dataloaders=loader)`.

    Each `__next__` returns a batch of up to `sequence_length` frames as a single
    `UnlabeledBatchDict`:

        frames     : (F, 3, H, W) float32 on GPU, ImageNet-normalized
        transforms : tensor([-1.]) sentinel meaning "no affine transform to undo"
        bbox       : (F, 4) full-frame
        is_multiview : False
    """

    def __init__(
        self,
        filename: str | Path,
        sequence_length: int,
        resize_dims: list[int] | None,
        device_id: int = 0,
        normalization_mean: list[float] = _IMAGENET_MEAN,
        normalization_std: list[float] = _IMAGENET_STD,
        num_iters: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        """Initialize the NVDEC-backed video iterator.

        Args:
            filename: absolute path to the video file
            sequence_length: number of frames per yielded batch
            resize_dims: [height, width] target size, or None for no resize
            device_id: CUDA device index for NVDEC and tensor placement
            normalization_mean: per-channel mean in [0, 1]
            normalization_std: per-channel std
            num_iters: total number of batches (computed if None)
            buffer_size: ThreadedDecoder prefetch buffer in frames; defaults to
                3 * sequence_length (NVIDIA-recommended 2-3x batch size)
        """
        nvc = _import_pynvc()

        self.filename = str(filename)
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f'{self.filename} is not a video file!')

        self.sequence_length = sequence_length
        self.resize_dims = resize_dims
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        # parity with LitDaliWrapper for downstream code that introspects loaders
        self.eval_mode = 'predict'
        self.do_context = False
        self.batch_sampler = 1

        # mean/std as (1, 3, 1, 1) on device for broadcast over (F, C, H, W)
        self._mean = torch.tensor(
            normalization_mean, device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            normalization_std, device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)

        if buffer_size is None:
            buffer_size = max(8, 3 * sequence_length)

        # planar RGB → frames arrive as (3, H, W) uint8 on GPU, no permute needed.
        self._decoder = nvc.ThreadedDecoder(
            enc_file_path=self.filename,
            buffer_size=buffer_size,
            gpu_id=self.device_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,
        )

        meta = self._decoder.get_stream_metadata()
        self._frame_count = int(getattr(meta, 'num_frames', 0))
        if self._frame_count <= 0:
            # not all containers report frame count up front; fall back to a high
            # estimate so num_iters doesn't cut off prematurely
            self._frame_count = 10 ** 12
        if num_iters is None:
            num_iters = int(np.ceil(self._frame_count / self.sequence_length))
        self.num_iters = num_iters

        self._exhausted = False
        # ThreadedDecoder is single-pass; on a second __iter__ we must reconfigure
        self._needs_reconfigure = False

    def __len__(self) -> int:
        return self.num_iters

    def __iter__(self) -> 'LitNvCodecWrapper':
        if self._needs_reconfigure:
            torch.cuda.current_stream().synchronize()
            self._decoder.reconfigure_decoder(self.filename)
            self._exhausted = False
        self._needs_reconfigure = True
        return self

    def __next__(self) -> UnlabeledBatchDict:
        if self._exhausted:
            raise StopIteration

        # background thread has already decoded these; returns immediately.
        frames = self._decoder.get_batch_frames(self.sequence_length)
        if len(frames) == 0:
            self._exhausted = True
            raise StopIteration

        # zero-copy DLPack views, then stack into a contiguous (F, 3, H, W) tensor.
        # stack copies into fresh storage, releasing the decoder's ring buffers
        # for the next prefetch.
        batch = torch.stack(
            [torch.from_dlpack(f) for f in frames],
            dim=0,
        )

        # batch is (F, 3, H, W) uint8 on GPU
        batch = batch.to(dtype=torch.float32) / 255.0

        if self.resize_dims is not None:
            batch = torch.nn.functional.interpolate(
                batch,
                size=(self.resize_dims[0], self.resize_dims[1]),
                mode='bilinear',
                align_corners=False,
            )

        batch = (batch - self._mean) / self._std

        height, width = batch.shape[-2], batch.shape[-1]
        bbox = torch.tensor(
            [0, 0, height, width], device=self.device, dtype=torch.float32,
        ).repeat(batch.shape[0], 1)

        # scalar sentinel matches DALI's "no transform" convention; consumed by
        # undo_affine_transform_batch
        transforms = torch.tensor([-1], device=self.device, dtype=torch.float32)

        # short batch = end of stream; next call will hit StopIteration
        if len(frames) < self.sequence_length:
            self._exhausted = True

        return UnlabeledBatchDict(
            frames=batch,
            transforms=transforms,
            bbox=bbox,
            is_multiview=False,
        )


class PrepareNvCodec:
    """Mirror of `lightning_pose.data.dali.PrepareDALI` backed by PyNvVideoCodec.

    Only the inference path (`train_stage="predict"`, `model_type="base"`,
    single-video, single-view) is implemented. Other configurations should be
    routed to `PrepareDALI` by the caller.
    """

    def __init__(
        self,
        train_stage: Literal['predict', 'train'],
        model_type: Literal['base', 'context'],
        filenames: list[str] | list[list[str]],
        resize_dims: list[int],
        dali_config: dict | DictConfig | None = None,
        imgaug: str | None = 'default',
        num_threads: int = 1,
    ) -> None:
        if train_stage != 'predict':
            raise NotImplementedError(
                'PrepareNvCodec only supports train_stage="predict"; '
                'use PrepareDALI for training.'
            )
        if model_type != 'base':
            raise NotImplementedError(
                'PrepareNvCodec only supports model_type="base"; '
                'use PrepareDALI for context (5-frame) models.'
            )

        # accept the same `filenames` shapes as PrepareDALI but reject multiview
        if isinstance(filenames, list) and isinstance(filenames[0], list):
            if len(filenames) > 1:
                raise NotImplementedError(
                    'PrepareNvCodec does not yet support multiview; use PrepareDALI.'
                )
            file_list = filenames[0]
        elif isinstance(filenames, list) and isinstance(filenames[0], str):
            file_list = filenames
        else:
            raise TypeError(f'unexpected filenames type: {type(filenames)}')

        if len(file_list) != 1:
            raise NotImplementedError(
                'PrepareNvCodec currently supports a single video at a time; '
                f'received {len(file_list)}.'
            )
        if not os.path.isfile(file_list[0]):
            raise FileNotFoundError(f'{file_list[0]} is not a video file!')

        self.train_stage = train_stage
        self.model_type = model_type
        self.filename = file_list[0]
        self.resize_dims = resize_dims
        self.dali_config = dali_config
        self.imgaug = imgaug
        self.num_threads = num_threads

        # respect DALI's per-stage sequence_length so batching is comparable
        self._sequence_length = self._resolve_sequence_length()

    def _resolve_sequence_length(self) -> int:
        """Pull `sequence_length` from `cfg.dali.base.predict` if present, else default to 16."""
        default = 16
        if self.dali_config is None:
            return default
        try:
            return int(self.dali_config['base']['predict']['sequence_length'])
        except (KeyError, TypeError):
            return default

    @property
    def num_iters(self) -> int:
        """Best-effort batch count; the loader itself stops on real EOF."""
        # We don't read frame count here to keep PyNvVideoCodec import lazy.
        # Lightning treats StopIteration as authoritative.
        return 10 ** 9

    def __call__(self) -> LitNvCodecWrapper:
        device_id = int(os.environ.get('LOCAL_RANK', '0'))
        return LitNvCodecWrapper(
            filename=self.filename,
            sequence_length=self._sequence_length,
            resize_dims=self.resize_dims,
            device_id=device_id,
        )
