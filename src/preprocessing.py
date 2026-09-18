"""Image preprocessing module for AI Document Verification System."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None


class ImagePreprocessor:
    """Reusable, configurable image preprocessing pipeline for documents."""

    def __init__(
        self,
        target_max_dim: int = 1200,
        apply_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.target_max_dim = target_max_dim
        self.apply_clahe = apply_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    @staticmethod
    def load_image(image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """Safely load an image from file path, bytes, PIL Image, or numpy array.

        Returns:
            RGB numpy array of shape (H, W, 3) and dtype uint8.
        """
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                if cv2 is not None:
                    return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                return np.stack([image_input] * 3, axis=-1)
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    if cv2 is not None:
                        return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                    return image_input[:, :, :3]
                return image_input
            raise ValueError(f"Unexpected image array shape: {image_input.shape}")

        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            return np.array(pil_img, dtype=np.uint8)

        if isinstance(image_input, (str, Path)):
            path = Path(image_input).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image not found at path: {path}")
            with Image.open(path) as im:
                pil_img = im.convert("RGB")
                return np.array(pil_img, dtype=np.uint8)

        if isinstance(image_input, bytes):
            import io
            with Image.open(io.BytesIO(image_input)) as im:
                pil_img = im.convert("RGB")
                return np.array(pil_img, dtype=np.uint8)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def resize_keep_aspect(self, image: np.ndarray, max_dim: Optional[int] = None) -> np.ndarray:
        """Resize image keeping aspect ratio such that max(height, width) <= max_dim."""
        max_dim = max_dim or self.target_max_dim
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image

        scale = max_dim / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if cv2 is not None:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        else:
            pil_img = Image.fromarray(image)
            resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            return np.array(resized, dtype=np.uint8)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to single-channel uint8 grayscale."""
        if image.ndim == 2:
            return image
        if cv2 is not None:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Standard ITU-R 601-2 luma transform
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def enhance_contrast(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        if not self.apply_clahe or cv2 is None:
            return gray_image
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(gray_image)

    @staticmethod
    def denoise(gray_image: np.ndarray) -> np.ndarray:
        """Apply mild Gaussian blur for noise reduction while preserving text edges."""
        if cv2 is not None:
            return cv2.GaussianBlur(gray_image, (3, 3), 0)
        return gray_image

    @staticmethod
    def binarize(gray_image: np.ndarray) -> np.ndarray:
        """Apply Otsu adaptive thresholding for clear foreground/background separation."""
        if cv2 is not None:
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        thresh = np.mean(gray_image)
        return ((gray_image > thresh) * 255).astype(np.uint8)

    def preprocess_for_ocr(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """Complete preprocessing pipeline optimized for OCR reading.

        Returns:
            Enhanced, cleaned grayscale image ready for Tesseract OCR.
        """
        rgb = self.load_image(image_input)
        resized = self.resize_keep_aspect(rgb)
        gray = self.to_grayscale(resized)
        enhanced = self.enhance_contrast(gray)
        denoised = self.denoise(enhanced)
        return denoised

    def preprocess_for_features(
        self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares both RGB and Grayscale representations for feature extraction.

        Returns:
            Tuple of (resized_rgb, enhanced_gray)
        """
        rgb = self.load_image(image_input)
        resized_rgb = self.resize_keep_aspect(rgb)
        gray = self.to_grayscale(resized_rgb)
        enhanced_gray = self.enhance_contrast(gray)
        return resized_rgb, enhanced_gray
