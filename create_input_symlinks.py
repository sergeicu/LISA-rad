import os
import argparse

def create_input_symlinks(inputfolder, labelsfolder, inputfolder_symlink):
    """Create symlinks in inputfolder_symlink for files that exist in labelsfolder."""
    if not os.path.exists(inputfolder_symlink):
        os.makedirs(inputfolder_symlink)

    for filename in os.listdir(inputfolder):
        input_file_path = os.path.join(inputfolder, filename)
        if not os.path.isfile(input_file_path):
            continue  # Skip if not a file

        name, ext = os.path.splitext(filename)
        label_file_path = os.path.join(labelsfolder, name + '.png')  # Assuming label files are .png

        if os.path.islink(label_file_path):
            target_path = os.path.join(inputfolder_symlink, filename)
            
            if os.path.exists(target_path):
                print(f"Symlink already exists: {target_path}")
            else:
                os.symlink(input_file_path, target_path)
                print(f"Created symlink: {target_path}")
        else:
            print(f"Skipping file (no corresponding label): {filename}")

def main():
    parser = argparse.ArgumentParser(description="Create symlinks in inputfolder_symlink for files that exist in labelsfolder.")
    parser.add_argument("inputfolder", help="Path to the input folder containing original files")
    parser.add_argument("labelsfolder", help="Path to the folder containing label symlinks")
    parser.add_argument("inputfolder_symlink", help="Path to the folder where input symlinks will be created")

    args = parser.parse_args()

    create_input_symlinks(args.inputfolder, args.labelsfolder, args.inputfolder_symlink)

if __name__ == "__main__":
    main()