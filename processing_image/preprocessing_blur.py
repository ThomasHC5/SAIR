import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from skimage import filters, img_as_ubyte

class FoveaBlur:
    def __init__(self, h, w, kernel_size, decay_rate=4.0):
        self.kernel_size = kernel_size
        center_x, center_y = w // 2, h // 2
        y_grid, x_grid = np.indices((h, w))
        dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = np.clip(dist / (max_dist + 1e-8), 0, 1)
        self.mask = np.exp(-decay_rate * norm_dist)
        self.mask = self.mask[:, :, np.newaxis]

    def __call__(self, img_np, dynamic_kernel_size=None):
        k_size = dynamic_kernel_size if dynamic_kernel_size is not None else self.kernel_size
        k_size = int(k_size)
        if k_size % 2 == 0: k_size += 1
        if k_size < 1: k_size = 1

        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
        blur_float = filters.gaussian(img_np, sigma=sigma, channel_axis=-1)
        blur_float = np.clip(blur_float, 0, 1)
        img_blur = img_as_ubyte(blur_float)
        img_original = img_np.astype(np.float32)
        img_blurred = img_blur.astype(np.float32)
        
        img_fovea = self.mask * img_original + (1 - self.mask) * img_blurred
        
        return img_fovea.astype(np.uint8)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def load_image(path: Path) -> np.ndarray:
    """Load an image and convert it to a NumPy array (RGB)."""
    return np.array(Image.open(path).convert('RGB'))

def process_image_with_blur(np_img: np.ndarray, kernel_size: int, decay_rate: float) -> Image.Image:
    """
    Apply Fovea Blur detection to the image.
    """
    h, w, _ = np_img.shape
    blur_op = FoveaBlur(h, w, kernel_size, decay_rate)
    blur_np = blur_op(np_img)
    return Image.fromarray(blur_np)

def process_directory(input_dir: str, output_dir: str, kernel_size: int, decay_rate: float, 
                      valid_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Recursively traverse the input directory, process images, and save them
    while maintaining the original directory structure.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    all_files = list(input_path.rglob('*'))
    img_paths = [p for p in all_files if p.suffix.lower() in valid_exts]

    print(f"Found {len(img_paths)} images in {input_path}")
    
    if not img_paths:
        print("No images found. Please check the input path.")
        return

    for p in tqdm(img_paths, desc="Processing Blur"):
        try:
            np_img = load_image(p)
            
            blur_pil = process_image_with_blur(np_img, kernel_size, decay_rate)

            relative_path = p.relative_to(input_path)
            save_dir = output_path / relative_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            save_name = f"{p.stem}_blur.jpg"
            blur_pil.save(save_dir / save_name)

        except Exception as e:
            print(f"Error processing file {p.name}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image Fovea Blur Preprocessing Script")
    parser.add_argument('--input_dir', type=str, 
                        default='../Data/Things-EEG2/Image_set/image_set/', 
                        help='Root directory of input images')
    parser.add_argument('--output_dir', type=str, 
                        default='../Data/Things-EEG2/Image_set/blur_set/', 
                        help='Directory to save processed images')
    parser.add_argument('--seed', type=int, default=20250908, help='Random seed')
    
    parser.add_argument('--kernel_size', type=int, default=21, 
                        help='Blur kernel size (must be odd), larger means more blurry background')
    parser.add_argument('--decay_rate', type=float, default=4.0, 
                        help='Rate of clarity decay from center (lambda in paper)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print("=== Starting Fovea Blur Preprocessing ===")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Parameters: Kernel Size={args.kernel_size}, Decay Rate={args.decay_rate}")
    
    set_seed(args.seed)
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        kernel_size=args.kernel_size,
        decay_rate=args.decay_rate
    )
    print("\n=== Processing Complete ===")