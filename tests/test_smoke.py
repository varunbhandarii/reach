"""Import checks for the core dependencies. Needs no camera, microphone or network."""

import importlib
import sys

import pytest

MODULES = [
    "numpy",
    "cv2",
    "mediapipe",
    "sounddevice",
    "openai",
    "websockets",
    "dotenv",
    "yaml",
    "meta_sam_parser",
]


@pytest.mark.parametrize("name", MODULES)
def test_core_dependency_imports(name):
    importlib.import_module(name)


def test_opencv_is_4x_with_kcf_tracker():
    import cv2

    assert cv2.__version__.startswith("4.")
    legacy = getattr(cv2, "legacy", None)
    assert hasattr(cv2, "TrackerKCF_create") or hasattr(legacy, "TrackerKCF_create")


def test_mediapipe_hand_landmarker_available():
    from mediapipe.tasks.python import vision

    assert hasattr(vision, "HandLandmarker")
    assert hasattr(vision, "HandLandmarkerOptions")


@pytest.mark.skipif(sys.platform != "win32", reason="pyttsx3 is only installed on Windows")
def test_windows_speech_dependency():
    importlib.import_module("pyttsx3")


def test_package_imports():
    import reach

    assert reach.__version__
