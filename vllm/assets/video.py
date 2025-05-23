# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import lru_cache
<<<<<<< HEAD
from typing import List, Literal
=======
from typing import ClassVar, Literal, Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import cv2
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from PIL import Image

<<<<<<< HEAD
from vllm.multimodal.video import sample_frames_from_video

from .base import get_cache_dir

=======
from vllm.utils import PlaceholderModule

from .base import get_cache_dir

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

@lru_cache
def download_video_asset(filename: str) -> str:
    """
    Download and open an image from huggingface
    repo: raushan-testing-hf/videos-test
    """
    video_directory = get_cache_dir() / "video-example-data"
    video_directory.mkdir(parents=True, exist_ok=True)

    video_path = video_directory / filename
    video_path_str = str(video_path)
    if not video_path.exists():
        video_path_str = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test",
            filename=filename,
            repo_type="dataset",
            cache_dir=video_directory,
        )
    return video_path_str


def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
<<<<<<< HEAD
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    frames = sample_frames_from_video(frames, num_frames)
=======

    num_frames = num_frames if num_frames > 0 else total_frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in range(total_frames):
        ok = cap.grab()  # next img
        if not ok:
            break
        if idx in frame_indices:  # only decompress needed
            ret, frame = cap.retrieve()
            if ret:
                frames.append(frame)

    frames = np.stack(frames)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


def video_to_pil_images_list(path: str,
<<<<<<< HEAD
                             num_frames: int = -1) -> List[Image.Image]:
=======
                             num_frames: int = -1) -> list[Image.Image]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    frames = video_to_ndarrays(path, num_frames)
    return [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]


<<<<<<< HEAD
@dataclass(frozen=True)
class VideoAsset:
    name: Literal["sample_demo_1.mp4"]
    num_frames: int = -1

    @property
    def pil_images(self) -> List[Image.Image]:
        video_path = download_video_asset(self.name)
=======
VideoAssetName = Literal["baby_reading"]


@dataclass(frozen=True)
class VideoAsset:
    name: VideoAssetName
    num_frames: int = -1

    _NAME_TO_FILE: ClassVar[dict[VideoAssetName, str]] = {
        "baby_reading": "sample_demo_1.mp4",
    }

    @property
    def filename(self) -> str:
        return self._NAME_TO_FILE[self.name]

    @property
    def pil_images(self) -> list[Image.Image]:
        video_path = download_video_asset(self.filename)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        ret = video_to_pil_images_list(video_path, self.num_frames)
        return ret

    @property
    def np_ndarrays(self) -> npt.NDArray:
<<<<<<< HEAD
        video_path = download_video_asset(self.name)
        ret = video_to_ndarrays(video_path, self.num_frames)
        return ret
=======
        video_path = download_video_asset(self.filename)
        ret = video_to_ndarrays(video_path, self.num_frames)
        return ret

    def get_audio(self, sampling_rate: Optional[float] = None) -> npt.NDArray:
        """
        Read audio data from the video asset, used in Qwen2.5-Omni examples.
        
        See also: examples/offline_inference/qwen2_5_omni/only_thinker.py
        """
        video_path = download_video_asset(self.filename)
        return librosa.load(video_path, sr=sampling_rate)[0]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
