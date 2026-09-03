import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ImageDataset(Dataset):
    """
    Dataset class to handle image loading and preprocessing.
    Returns the preprocessed tensor and the original file path.
    """
    def __init__(self, image_paths: List[Path], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            image = self.preprocess(image)
            return image, str(path)
        except Exception as e:
            # Return None to filter out corrupt images in collate_fn if needed
            # For simplicity, we just print and return a dummy tensor here (or handle externally)
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 224, 224), str(path)

def extract_and_save(loader, model, input_root, output_root, device):
    """
    Run inference in batches and save embeddings preserving directory structure.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Processing Batches"):
            images = images.to(device)
            
            # Encode images using the model
            features = model.encode_image(images)
            features = features.cpu().numpy()

            # Save each feature in the batch
            for i, path_str in enumerate(paths):
                original_path = Path(path_str)
                
                # Skip if image was dummy (failed to load)
                if images[i].sum() == 0 and features[i].sum() == 0: 
                    continue

                # Maintain directory structure
                relative_path = original_path.relative_to(input_root)
                save_dir = output_root / relative_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / f"{original_path.stem}.npy"
                np.save(save_path, features[i])

def main():
    parser = argparse.ArgumentParser(description="CLIP Embedding Preprocessing")
    parser.add_argument('--input_dir', type=str, default='Data/Things-EEG2/Image_set/edge_set/', help='Root directory of input images')
    parser.add_argument('--output_dir', type=str, default='Data/Things-EEG2/Features_all/edge/', help='Directory to save processed embeddings')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device: cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=20250908, help='Random seed')
    parser.add_argument('--model', type=str, default='hf-hub:thaottn/OpenCLIP-resnet50-CC12M', help='OpenCLIP model')
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 1. Setup Model
    print(f"Loading model {args.model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    model.to(args.device)
    model.eval()

    # 2. Find Images
    input_path = Path(args.input_dir)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_files = list(input_path.rglob('*'))
    img_paths = [p for p in all_files if p.suffix.lower() in valid_exts]
    
    print(f"Found {len(img_paths)} images in {args.input_dir}")
    if not img_paths:
        return

    # 3. Create DataLoader for Batch Processing
    dataset = ImageDataset(img_paths, preprocess)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 4. Run Inference and Save
    print(f"Starting inference on {args.device} with batch size {args.batch_size}...")
    extract_and_save(loader, model, args.input_dir, args.output_dir, args.device)
    print("Processing Complete.")

if __name__ == '__main__':
    main()