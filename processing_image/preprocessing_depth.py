import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm
import matplotlib
import cv2

import torch
from depth_anything_v2.dpt import DepthAnythingV2

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_directory(input_dir: str, output_dir: str, device: str, valid_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Recursively traverse the input directory, process images, and save them
    while maintaining the original directory structure.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Initialize model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()

    # Recursively find all files in the input directory
    all_files = list(input_path.rglob('*'))
    img_paths = [p for p in all_files if p.suffix.lower() in valid_exts]

    print(f"Found {len(img_paths)} images in {input_path}")
    
    if not img_paths:
        print("No images found. Please check the input path.")
        return

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # with torch.no_grad():
    for img_path in tqdm(img_paths, desc="Processing"):

        # try:
        # Inference
        raw_img = cv2.imread(str(img_path))
        depth = model.infer_image(
            raw_img
        )

        # Save results
        relative_path = img_path.relative_to(input_path)
        save_dir = output_path / relative_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
            
        # Visualize and Save
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        save_name = f"{img_path.stem}_depth.png"
        cv2.imwrite(save_dir / save_name, depth)

        # except Exception as e:
        #     print(f"Error processing batch starting at index {i}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Depth Estimation Preprocessing")
    parser.add_argument('--input_dir', type=str, 
                        default='../Data/Things-EEG2/Image_set/image_set/', 
                        help='Root directory of input images')
    parser.add_argument('--output_dir', type=str, 
                        default='../Data/Things-EEG2/Image_set/depth_set/', 
                        help='Directory to save processed images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device: cpu or cuda')
    parser.add_argument('--seed', type=int, default=20250908, help='Random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print("=== Starting Depth Estimation ===")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    
    set_seed(args.seed)
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    print("\n=== Processing Complete ===")