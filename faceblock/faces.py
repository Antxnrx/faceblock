"""Stage 1 - face detection and encoding.

Uses OpenCV's bundled YuNet detector and SFace recogniser (ONNX models from the
OpenCV Zoo). Chosen over dlib/DeepFace because both models are small, run on CPU
in milliseconds, and need no compilation toolchain - which keeps `pip install`
reproducible on a clean machine during a live demo.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from .config import MODEL_DIR

# NOTE: opencv_zoo stores model weights in Git LFS, so raw.githubusercontent.com
# serves a ~130-byte pointer file rather than the model. The media.* host
# resolves LFS objects to their real contents.
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
DETECTOR_MODEL = (
    f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2023mar.onnx",
    232589,
)
RECOGNIZER_MODEL = (
    f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "face_recognition_sface_2021dec.onnx",
    38696353,
)


@dataclass
class Face:
    """One detected face: where it is, how confident the detector was, and its embedding."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    detector_score: float
    embedding: np.ndarray  # 128-d, L2-normalised so dot product == cosine similarity

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


def _ensure_model(url: str, filename: str, expected_size: int) -> Path:
    """Download a model into the cache directory on first use, verifying its size."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    if path.exists() and path.stat().st_size == expected_size:
        return path

    print(f"  downloading model {filename} ({expected_size / 1e6:.1f} MB) ...")
    response = requests.get(url, timeout=180, stream=True)
    response.raise_for_status()
    # Write to a temp path first so an interrupted download never leaves a
    # truncated model that would fail confusingly on the next run.
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 16):
            handle.write(chunk)

    actual = tmp.stat().st_size
    if actual != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Model download for {filename} returned {actual} bytes, expected "
            f"{expected_size}. A ~130-byte result means the Git LFS pointer was "
            f"served instead of the model. Source: {url}"
        )
    tmp.replace(path)
    return path


_detector = None
_recognizer = None

# OpenCV's FaceDetectorYN/FaceRecognizerSF are NOT thread-safe: detect() reads
# input dimensions that setInputSize() mutates on the shared instance, so
# concurrent calls segfault rather than failing cleanly. Candidate images are
# fetched in parallel, so inference has to be serialised behind this lock.
# Detection takes milliseconds while the downloads take seconds, so the lost
# parallelism costs effectively nothing.
_model_lock = threading.Lock()


def _get_models() -> tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    global _detector, _recognizer
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(_ensure_model(*DETECTOR_MODEL)), "", (320, 320), 0.85, 0.3, 5000
        )
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(
            str(_ensure_model(*RECOGNIZER_MODEL)), ""
        )
    return _detector, _recognizer


def load_image(source: str | Path | bytes) -> np.ndarray:
    """Decode an image from a path or raw bytes into a BGR array."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    else:
        data = source

    array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if array is not None:
        return array

    # OpenCV can't decode every format it's handed (some WebP/AVIF variants);
    # Pillow covers the rest, and matters because candidate images are fetched
    # from arbitrary sites whose formats we don't control.
    with Image.open(io.BytesIO(data)) as img:
        rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_faces(image: np.ndarray) -> list[Face]:
    """Detect and encode every face in a BGR image, largest first."""
    faces: list[Face] = []
    with _model_lock:
        detector, recognizer = _get_models()
        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, raw = detector.detect(image)
        if raw is None:
            return []

        for row in raw:
            aligned = recognizer.alignCrop(image, row)
            embedding = recognizer.feature(aligned).flatten().astype(np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm == 0:
                continue  # degenerate crop; nothing meaningful to compare against
            x, y, w, h = (int(v) for v in row[:4])
            faces.append(
                Face(
                    bbox=(x, y, w, h),
                    detector_score=float(row[-1]),
                    embedding=embedding / norm,
                )
            )

    faces.sort(key=lambda f: f.area, reverse=True)
    return faces


def primary_face(image: np.ndarray) -> Face | None:
    """The largest detected face - the subject of the photo, by convention."""
    faces = detect_faces(image)
    return faces[0] if faces else None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings."""
    return float(np.dot(a, b))
