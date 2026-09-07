"""HYPIR image upscaling wrapper for the townsfolk SAM3D pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
HYPIR_ROOT = REPO_ROOT / "HYPIR"
DEFAULT_WEIGHT_PATH = REPO_ROOT / "checkpoints/hypir/HYPIR_sd2.pth"
DEFAULT_BASE_MODEL = str(REPO_ROOT / "checkpoints/sd2-1-base")
DEFAULT_UPSCALE = 4
DEFAULT_PROMPT = "isometric pixel art game character sprite, sharp details"
DEFAULT_SEED = 231

LORA_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv_out",
    "proj_in",
    "proj_out",
    "ff.net.2",
    "ff.net.0.proj",
]


class HypirUpscaler:
    """Lazy-loaded HYPIR SD2 enhancer for batch frame upscaling."""

    def __init__(
        self,
        weight_path: str | Path = DEFAULT_WEIGHT_PATH,
        base_model_path: str = DEFAULT_BASE_MODEL,
        device: str = "cuda",
        upscale: int = DEFAULT_UPSCALE,
        prompt: str = DEFAULT_PROMPT,
        patch_size: int = 512,
        stride: int = 256,
        seed: int = DEFAULT_SEED,
    ):
        self.weight_path = Path(weight_path)
        self.base_model_path = base_model_path
        self.device = device
        self.upscale = upscale
        self.prompt = prompt
        self.patch_size = patch_size
        self.stride = stride
        self.seed = seed
        self._model = None

    def _ensure_hypir_path(self) -> None:
        hypir_path = str(HYPIR_ROOT)
        if hypir_path not in sys.path:
            sys.path.insert(0, hypir_path)

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"HYPIR weights not found at {self.weight_path}. "
                "Download with: hf download lxq007/HYPIR HYPIR_sd2.pth --local-dir checkpoints/hypir"
            )

        self._ensure_hypir_path()
        from accelerate.utils import set_seed
        from HYPIR.enhancer.sd2 import SD2Enhancer

        set_seed(self.seed)
        print(f"Loading HYPIR from {self.weight_path}...")
        self._model = SD2Enhancer(
            base_model_path=self.base_model_path,
            weight_path=str(self.weight_path),
            lora_modules=LORA_MODULES,
            lora_rank=256,
            model_t=200,
            coeff_t=200,
            device=self.device,
        )
        self._model.init_models()
        print("HYPIR loaded.")

    def upscale_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """Upscale a BGR uint8 image using HYPIR."""
        self.load()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_tensor = to_tensor(pil_image).unsqueeze(0)

        result = self._model.enhance(
            lq=image_tensor,
            prompt=self.prompt,
            scale_by="factor",
            upscale=self.upscale,
            patch_size=self.patch_size,
            stride=self.stride,
            return_type="pil",
        )[0]

        upscaled_rgb = np.array(result.convert("RGB"))
        return cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)

    def upscale_file(self, input_path: Path, output_path: Path | None = None) -> np.ndarray:
        image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {input_path}")

        upscaled = self.upscale_bgr(image_bgr)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), upscaled)
        return upscaled
