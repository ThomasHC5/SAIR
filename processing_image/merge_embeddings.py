import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def merge_npy_files(input_dir: str, output_path: str, save_filenames: bool = False):
    """
    Loads all .npy files from input_dir, sorts them, and merges them into a single file.
    """
    root = Path(input_dir)
    # 1. Find and Sort files
    # Using sorted() ensures deterministic order (usually alphabetical)
    print(f"Scanning {root} for .npy files...")
    npy_files = sorted(list(root.rglob('*.npy')))
    
    if not npy_files:
        print("No .npy files found.")
        return

    print(f"Found {len(npy_files)} files. Loading...")

    # 2. Load data
    data_list = []
    file_names = []
    
    for p in tqdm(npy_files, desc="Merging"):
        # Load the feature vector
        feat = np.load(p)
        
        # Ensure it's 2D (1, Dimensions) for vstack, or flatten if needed
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
            
        data_list.append(feat)
        
        # Keep track of the filename/path relative to input_dir
        # This is crucial to know which row corresponds to which image
        file_names.append(str(p.relative_to(root)))

    # 3. Stack into a single matrix
    # Shape will be (N_images, Embedding_Dimension)
    all_features = np.vstack(data_list)
    print(f"Final shape: {all_features.shape}")

    # 4. Save
    # Save the big feature matrix
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_features)
    print(f"Saved merged features to {output_path}")

    # Optionally save the order of files
    if save_filenames:
        meta_path = Path(output_path).parent / "filenames.npy"
        np.save(meta_path, np.array(file_names))
        print(f"Saved corresponding filenames to {meta_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge NPY files")
    parser.add_argument('--input_dir', type=str, default='Data/Things-EEG2/Features_all/blur/training_images/', help='Directory containing split .npy files')
    parser.add_argument('--output_path', type=str, default='Data/Things-EEG2/Features/blur_training.npy', help='Path for the final merged .npy file')
    
    args = parser.parse_args()
    
    merge_npy_files(args.input_dir, args.output_path)