"""Video loading via NVIDIA PyNvVideoCodec (direct NVDEC), as an alternative to DALI.

This module provides a drop-in replacement for the DALI-based prediction loader,
producing the same `UnlabeledBatchDict` / `MultiviewUnlabeledBatchDict` outputs that
`predict_step` consumes. It is designed for inference only (`train_stage="predict"`).

Single-video, single-view, base (non-context) models are supported. Context (5-frame)
and multi-view modes raise `NotImplementedError` and should fall back to DALI.

Why this module:
    For workloads where decode is the bottleneck and the GPU has multiple NVDEC engines,
    PyNvVideoCodec gives finer-grained control over decoder placement, CUDA streams, and
    parallel decode across files than DALI's `fn.readers.video`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from omegaconf import DictConfig

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.datatypes import UnlabeledBatchDict
from lightning_pose.data.utils import count_frames

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

    Frames are produced on the GPU, resized (optional), scaled to [0, 1], normalized
    with ImageNet statistics, and reshaped to `(sequence_length, 3, H, W)` matching
    DALI's `crop_mirror_normalize(output_layout="FCHW")`.
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
        """
        nvc = _import_pynvc()

        self.filename = str(filename)
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f'{self.filename} is not a video file!')

        self.sequence_length = sequence_length
        self.resize_dims = resize_dims
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.eval_mode = 'predict'
        self.do_context = False
        self.batch_sampler = 1  # parity with LitDaliWrapper hack

        # mean/std as (1, 3, 1, 1) on device for broadcast over (F, C, H, W)
        self._mean = torch.tensor(
            normalization_mean, device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            normalization_std, device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self._frame_count = count_frames(self.filename)
        if num_iters is None:
            num_iters = int(np.ceil(self._frame_count / self.sequence_length))
        self.num_iters = num_iters

        # SimpleDecoder yields RGB frames on the device when use_device_memory=True.
        # Output tensors expose __cuda_array_interface__ so torch.as_tensor works.
        self._decoder = nvc.SimpleDecoder(
            enc_file_path=self.filename,
            gpu_id=self.device_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )
        self._frame_iter: Iterator | None = None
        self._exhausted = False

    def __len__(self) -> int:
        return self.num_iters

    def __iter__(self) -> 'LitNvCodecWrapper':
        # SimpleDecoder is itself iterable; reset state for a fresh pass
        self._frame_iter = iter(self._decoder)
        self._exhausted = False
        return self

    def _frame_to_tensor(self, frame) -> torch.Tensor:
        """Convert a SimpleDecoder frame (CAI-compatible) to a (3, H, W) float CUDA tensor.

        Args:
            frame: PyNvVideoCodec frame object exposing __cuda_array_interface__

        Returns:
            uint8 tensor of shape (H, W, 3) on the configured CUDA device
        """
        # zero-copy view onto the decoder's device buffer
        t = torch.as_tensor(frame, device=self.device)
        # SimpleDecoder RGB output is (H, W, 3) uint8
        if t.dim() == 3 and t.shape[-1] == 3:
            return t
        # some PyNvVideoCodec versions return (3, H, W); normalize to (H, W, 3)
        if t.dim() == 3 and t.shape[0] == 3:
            return t.permute(1, 2, 0).contiguous()
        raise RuntimeError(f'unexpected NVDEC frame shape: {tuple(t.shape)}')

    def __next__(self) -> UnlabeledBatchDict:
        if self._frame_iter is None:
            self.__iter__()
        if self._exhausted:
            raise StopIteration

        frames_hw3: list[torch.Tensor] = []
        for _ in range(self.sequence_length):
            try:
                f = next(self._frame_iter)  # type: ignore[arg-type]
            except StopIteration:
                self._exhausted = True
                break
            frames_hw3.append(self._frame_to_tensor(f))

        if not frames_hw3:
            raise StopIteration

        # stack -> (F, H, W, 3) uint8 on GPU, then -> (F, 3, H, W) float
        batch = torch.stack(frames_hw3, dim=0).permute(0, 3, 1, 2).contiguous()
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
        self.frame_count = count_frames(self.filename)

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
        return int(np.ceil(self.frame_count / self._sequence_length))

    def __call__(self) -> LitNvCodecWrapper:
        device_id = int(os.environ.get('LOCAL_RANK', '0'))
        return LitNvCodecWrapper(
            filename=self.filename,
            sequence_length=self._sequence_length,
            resize_dims=self.resize_dims,
            device_id=device_id,
            num_iters=self.num_iters,
        )
