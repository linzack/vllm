# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

import numpy.typing as npt

from vllm.utils import PlaceholderModule

from .base import VLLM_S3_BUCKET_URL, get_vllm_public_assets

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

ASSET_DIR = "multimodal_asset"

<<<<<<< HEAD

@dataclass(frozen=True)
class AudioAsset:
    name: Literal["winning_call", "mary_had_lamb"]

    @property
    def audio_and_sample_rate(self) -> tuple[npt.NDArray, float]:
        audio_path = get_vllm_public_assets(filename=f"{self.name}.ogg",
=======
AudioAssetName = Literal["winning_call", "mary_had_lamb"]


@dataclass(frozen=True)
class AudioAsset:
    name: AudioAssetName

    @property
    def filename(self) -> str:
        return f"{self.name}.ogg"

    @property
    def audio_and_sample_rate(self) -> tuple[npt.NDArray, float]:
        audio_path = get_vllm_public_assets(filename=self.filename,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                                            s3_prefix=ASSET_DIR)
        return librosa.load(audio_path, sr=None)

    def get_local_path(self) -> Path:
<<<<<<< HEAD
        return get_vllm_public_assets(filename=f"{self.name}.ogg",
=======
        return get_vllm_public_assets(filename=self.filename,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                                      s3_prefix=ASSET_DIR)

    @property
    def url(self) -> str:
        return urljoin(VLLM_S3_BUCKET_URL, f"{ASSET_DIR}/{self.name}.ogg")
