import os
import argparse
import numpy as np
import cv2

def has_values_less_than_10(png_path):
    """Check if the PNG file has any values less than 10."""
    image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    return np.any(image < 10)

def create_symlinks(source_folder, target_folder, remove_existing=False, skip_existing=False):
    """Create symlinks for PNG files with values less than 10."""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.png'):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            if has_values_less_than_10(source_path):
                if os.path.exists(target_path):
                    if remove_existing:
                        os.unlink(target_path)
                    elif skip_existing:
                        print(f"Skipping existing symlink: {target_path}")
                        continue
                    else:
                        print(f"Symlink already exists: {target_path}")
                        continue

                os.symlink(source_path, target_path)
                print(f"Created symlink: {target_path}")
            else:
                print(f"Skipping file (no values < 10): {source_path}")

def main():
    parser = argparse.ArgumentParser(description="Create symlinks for PNG files with values less than 10.")
    parser.add_argument("source_folder", help="Path to the source folder containing PNG files")
    parser.add_argument("target_folder", help="Path to the target folder for symlinks")
    parser.add_argument("--remove-existing", action="store_true", help="Remove existing symlinks before creating new ones")
    parser.add_argument("--skip-existing", action="store_true", help="Skip creating symlinks if they already exist")

    args = parser.parse_args()

    create_symlinks(args.source_folder, args.target_folder, args.remove_existing, args.skip_existing)

if __name__ == "__main__":
    main()