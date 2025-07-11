from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Optional

import cv2
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from PIL import Image

from vllm.utils import PlaceholderModule
from .base import get_cache_dir

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

VideoAssetName = Literal["baby_reading"]

@dataclass
class VideoConfig:
    """
    Configuration for frame extraction & resizing.
    All values have sensible defaults but can be overridden.
    """
    # Single-frame resize limits (pixels)
    min_pixels: int = 1
    max_pixels: int = 100176
    # Total pixel budget (all frames): default 90% of 128k*28*28
    total_pixels_limit: int = int(128000 * 28 * 28 * 0.9)
    # Frame sampling must be multiple of this
    frame_factor: int = 2
    # Default fps and limits when sampling by fps
    default_fps: float = 1.0
    fps_min_frames: int = 1
    fps_max_frames: int = 32


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return int(np.ceil(number / factor)) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return int(np.floor(number / factor)) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """
    Resize dimensions divisible by factor, within min/max pixel count,
    preserving aspect ratio.
    """
    # prevent extreme aspect ratios
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Aspect ratio too large")
    # initial rounding
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    # enforce max_pixels
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(int(height / beta), factor))
        w_bar = max(factor, floor_by_factor(int(width / beta), factor))
    # enforce min_pixels
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = max(factor, ceil_by_factor(int(height * beta), factor))
        w_bar = max(factor, ceil_by_factor(int(width * beta), factor))
    return h_bar, w_bar


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: float,
    cfg: VideoConfig,
) -> int:
    """
    Determine number of frames to sample: either fixed nframes or by fps.
    """
    assert not ("nframes" in ele and "fps" in ele), "Specify only nframes or fps"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], cfg.frame_factor)
    else:
        fps = ele.get("fps", cfg.default_fps)
        min_f = ele.get("min_frames", cfg.fps_min_frames)
        max_f = ele.get("max_frames", cfg.fps_max_frames)
        n = total_frames / video_fps * fps
        nframes = floor_by_factor(
            max(min(n, min(max_f, total_frames)), min_f),
            cfg.frame_factor,
        )
    if not (cfg.frame_factor <= nframes <= total_frames):
        raise ValueError(f"nframes should be in [{cfg.frame_factor}, {total_frames}], got {nframes}")
    return nframes


@lru_cache
def download_video_asset(filename: str) -> str:
    cache_dir = get_cache_dir() / "video-example-data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not path.exists():
        return hf_hub_download(
            repo_id="raushan-testing-hf/videos-test",
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    return str(path)


def _load_video_ndarrays(path: str) -> npt.NDArray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total):
        ok = cap.grab()
        if not ok:
            break
        ret, frame = cap.retrieve()
        if ret:
            frames.append(frame)
    cap.release()
    return np.stack(frames)


def _load_audio_array(path: str, sr: Optional[float]) -> np.ndarray:
    arr, _ = librosa.load(path, sr=sr)
    return arr


class VideoAsset:
    """
    Video + audio loader supporting name/path/ndarray inputs.
    """
    _NAME_TO_FILE: dict[VideoAssetName, str] = {"baby_reading": "sample_demo_1.mp4"}

    def __init__(
        self,
        *,
        name: Optional[VideoAssetName] = None,
        path: Optional[str] = None,
        num_frames: int = -1,
        video_ndarrays: Optional[npt.NDArray] = None,
        audio_array: Optional[np.ndarray] = None,
        audio_sr: Optional[float] = None,
        config: VideoConfig = VideoConfig(),
    ):
        if not (name or path or video_ndarrays is not None):
            raise ValueError("Provide name, path, or video_ndarrays")
        if name and path:
            raise ValueError("Only one of name or path allowed")
        self.name = name
        self.path = path
        self.num_frames = num_frames
        self._video_ndarrays = video_ndarrays
        self._audio_array = audio_array
        self._audio_sr = audio_sr
        self.config = config

    @property
    def _video_path(self) -> Optional[str]:
        if self.name:
            fname = self._NAME_TO_FILE[self.name]
            return download_video_asset(fname)
        return self.path

    @property
    def metadata(self) -> dict[str, Any]:
        p = self._video_path
        assert p, "No video path"
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {p}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return {"total_num_frames": total, "fps": fps}

    @property
    def np_ndarrays(self) -> npt.NDArray:
        # load raw frames
        if self._video_ndarrays is not None:
            raw = self._video_ndarrays
        else:
            p = self._video_path
            assert p, "No video path"
            raw = _load_video_ndarrays(p)
        total = raw.shape[0]
        # sample frame count
        if self.num_frames > 0:
            nframes = self.num_frames
        else:
            meta = self.metadata
            ele = {"fps": self.config.default_fps,
                   "min_frames": self.config.fps_min_frames,
                   "max_frames": self.config.fps_max_frames}
            nframes = smart_nframes(ele, meta["total_num_frames"], meta["fps"], self.config)
        idxs = np.linspace(0, total-1, nframes, dtype=int)
        # resize frames
        out = []
        for i in idxs:
            frm = raw[i]
            h, w = frm.shape[:2]
            nh, nw = smart_resize(h, w,
                                  factor=self.config.frame_factor,
                                  min_pixels=self.config.min_pixels,
                                  max_pixels=self.config.max_pixels)
            out.append(cv2.resize(frm, (nw, nh)))
        return np.stack(out)

    @property
    def pil_images(self) -> list[Image.Image]:
        return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in self.np_ndarrays]

    def get_audio(self, sampling_rate: Optional[float] = None) -> np.ndarray:
        if self._audio_array is not None:
            return self._audio_array
        p = self._video_path
        assert p, "No video path"
        return _load_audio_array(p, sampling_rate or self._audio_sr)
