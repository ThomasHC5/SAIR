import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import feature, color, img_as_ubyte
from tqdm import tqdm

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def load_image(path: Path) -> np.ndarray:
    """Load an image and convert it to a NumPy array (RGB)."""
    return np.array(Image.open(path).convert('RGB'))

def detect_edges(np_img: np.ndarray, sigma: float = 1.0) -> Image.Image:
    """
    Apply Canny edge detection to the image.
    
    Args:
        np_img: Input RGB image array.
        sigma: Standard deviation of the Gaussian filter for Canny.
    
    Returns:
        PIL.Image: Edge-detected image in RGB mode.
    """
    gray = color.rgb2gray(np_img)
    edges = feature.canny(gray, sigma=sigma)
    edges_u8 = img_as_ubyte(edges)
    return Image.fromarray(edges_u8).convert("RGB")

def process_directory(input_dir: str, output_dir: str, valid_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Recursively traverse the input directory, process images, and save them
    while maintaining the original directory structure.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Recursively find all files in the input directory
    all_files = list(input_path.rglob('*'))
    img_paths = [p for p in all_files if p.suffix.lower() in valid_exts]

    print(f"Found {len(img_paths)} images in {input_path}")
    
    if not img_paths:
        print("No images found. Please check the input path.")
        return

    for p in tqdm(img_paths, desc="Processing"):
        try:
            np_img = load_image(p)
            edge_pil = detect_edges(np_img)
            
            relative_path = p.relative_to(input_path)
            save_dir = output_path / relative_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            save_name = f"{p.stem}_edge.jpg"
            edge_pil.save(save_dir / save_name)

        except Exception as e:
            print(f"Error processing file {p.name}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image Edge Detection Preprocessing Script")
    parser.add_argument('--input_dir', type=str, 
                        default='../Data/Things-EEG2/Image_set/image_set/', 
                        help='Root directory of input images')
    parser.add_argument('--output_dir', type=str, 
                        default='../Data/Things-EEG2/Image_set/edge_set/', 
                        help='Directory to save processed images')
    parser.add_argument('--seed', type=int, default=20250908, help='Random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print("=== Starting Edge Detection Preprocessing ===")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    
    set_seed(args.seed)
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    print("\n=== Processing Complete ===")