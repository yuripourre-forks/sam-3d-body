"""Lightweight "Clarity-style" upscaler for the townsfolk SAM3D pipeline.

The real philz1337x/clarity-upscaler ships as a fork of the full AUTOMATIC1111
WebUI (SD1.5 checkpoints + ControlNet-tile + Tiled Diffusion/Tiled VAE,
packaged for Replicate's `cog`). Self-hosting it requires ~15-20GB of extra
model downloads, a second webui repo, and dependencies pinned to
`xformers==0.0.22` / CUDA-specific code paths that conflict with this
project's ROCm-based environment.

This module instead reproduces Clarity's *core technique* -- a traditional
upscale followed by a low-denoising-strength diffusion img2img refinement
pass -- using the same ROCm-compatible SD2.1 base checkpoint already
downloaded for HYPIR, so it can be run and compared fairly in this
environment. It intentionally mirrors HypirUpscaler's interface
(`load()` / `upscale_bgr()` / `upscale_file()`) so it is a drop-in
alternative in the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = str(REPO_ROOT / "checkpoints/sd2-1-base")
DEFAULT_UPSCALE = 4
DEFAULT_PROMPT = (
    "masterpiece, best quality, highres, isometric pixel art game character "
    "sprite, sharp details, crisp edges"
)
DEFAULT_NEGATIVE_PROMPT = "blurry, lowres, low quality, jpeg artifacts, distorted, deformed"
DEFAULT_DENOISE_STRENGTH = 0.35
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 6.0
DEFAULT_SEED = 1337
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 64


class ClarityStyleUpscaler:
    """Lazy-loaded SD2.1 img2img refiner approximating Clarity Upscaler's recipe."""

    def __init__(
        self,
        base_model_path: str = DEFAULT_BASE_MODEL,
        device: str = "cuda",
        upscale: int = DEFAULT_UPSCALE,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        denoising_strength: float = DEFAULT_DENOISE_STRENGTH,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        seed: int = DEFAULT_SEED,
    ):
        self.base_model_path = base_model_path
        self.device = device
        self.upscale = upscale
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.denoising_strength = denoising_strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.seed = seed
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return

        print(f"Loading Clarity-style SD2.1 img2img pipeline from {self.base_model_path}...")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        print("Clarity-style pipeline loaded.")

    def _refine_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        pil_tile = Image.fromarray(tile_rgb)
        result = self._pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=pil_tile,
            strength=self.denoising_strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]
        return np.array(result.convert("RGB").resize(pil_tile.size, Image.LANCZOS))

    def upscale_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """Upscale a BGR uint8 image: Lanczos resize + SD2.1 img2img detail refinement."""
        self.load()

        height, width = image_bgr.shape[:2]
        target_size = (width * self.upscale, height * self.upscale)
        resized_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        target_w, target_h = target_size
        if target_w <= self.tile_size and target_h <= self.tile_size:
            refined_rgb = self._refine_tile(resized_rgb)
        else:
            refined_rgb = self._refine_tiled(resized_rgb)

        return cv2.cvtColor(refined_rgb, cv2.COLOR_RGB2BGR)

    def _refine_tiled(self, image_rgb: np.ndarray) -> np.ndarray:
        """MultiDiffusion-style overlapping tile refinement with linear-ramp blending."""
        height, width = image_rgb.shape[:2]
        stride = self.tile_size - self.tile_overlap
        accum = np.zeros((height, width, 3), dtype=np.float32)
        weight = np.zeros((height, width, 1), dtype=np.float32)
        ramp = (
            np.linspace(0, 1, self.tile_overlap, dtype=np.float32)
            if self.tile_overlap > 0
            else np.array([], dtype=np.float32)
        )

        y0 = 0
        while True:
            x0 = 0
            while True:
                tile = image_rgb[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size]
                tile_h, tile_w = tile.shape[:2]
                refined = self._refine_tile(tile)

                tile_weight = np.ones((tile_h, tile_w, 1), dtype=np.float32)
                if self.tile_overlap > 0:
                    if x0 > 0:
                        tile_weight[:, : self.tile_overlap, 0] *= ramp
                    if y0 > 0:
                        tile_weight[: self.tile_overlap, :, 0] *= ramp[:, None]

                accum[y0 : y0 + tile_h, x0 : x0 + tile_w] += refined.astype(np.float32) * tile_weight
                weight[y0 : y0 + tile_h, x0 : x0 + tile_w] += tile_weight

                if x0 + self.tile_size >= width:
                    break
                x0 = min(x0 + stride, max(width - self.tile_size, 0))
            if y0 + self.tile_size >= height:
                break
            y0 = min(y0 + stride, max(height - self.tile_size, 0))

        weight = np.clip(weight, 1e-6, None)
        blended = accum / weight
        return np.clip(blended, 0, 255).astype(np.uint8)

    def upscale_file(self, input_path: Path, output_path: Path | None = None) -> np.ndarray:
        image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {input_path}")

        upscaled = self.upscale_bgr(image_bgr)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), upscaled)
        return upscaled
