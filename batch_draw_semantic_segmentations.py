import os
import argparse
import cv2
import numpy as np
from PIL import Image
import glob

def create_semantic_segmentation(image_path, txt_file_path,fracture_only=False):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Create a blank segmentation map (initialized with 0 for background)
    # segmentation_map = np.zeros((height, width), dtype=np.uint8)
    
    # Create a blank segmentation map with 255 for the background
    segmentation_map = np.full((height, width), 255, dtype=np.uint8)
    
    
    # Read the segmentation data from the txt file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            label = int(data[0]) 
            
            # skip all labels except fracture 
            if fracture_only:
                if label !=3:
                    continue
            x_center = float(data[1])
            y_center = float(data[2])
            seg_width = float(data[3])
            seg_height = float(data[4])
            
            # Convert normalized coordinates back to pixel coordinates
            x1 = int((x_center - seg_width / 2) * width)
            y1 = int((y_center - seg_height / 2) * height)
            x2 = int((x_center + seg_width / 2) * width)
            y2 = int((y_center + seg_height / 2) * height)
            
            # Fill the segmentation area with the class label
            segmentation_map[y1:y2, x1:x2] = label
    
    return segmentation_map

def create_colored_segmentation(segmentation_map):
    colors = [
        (0, 0, 0),      # Black (background)
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128)     # Navy
    ]
    
    height, width = segmentation_map.shape
    colored_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(colors):
        colored_map[segmentation_map == class_id] = color
    
    return colored_map

def save_image_with_reference(colored_segmentation, reference_image_path, output_path):
    # Read the reference image
    reference_image = cv2.imread(reference_image_path)
    
    # Ensure both images have the same height
    height = max(reference_image.shape[0], colored_segmentation.shape[0])
    width_ref = reference_image.shape[1]
    width_seg = colored_segmentation.shape[1]
    
    # If the heights differ, resize the images
    if reference_image.shape[0] != height:
        reference_image = cv2.resize(reference_image, (width_ref, height))
    if colored_segmentation.shape[0] != height:
        colored_segmentation = cv2.resize(colored_segmentation, (width_seg, height))
    
    # Concatenate the two images side by side
    concatenated_image = np.hstack((reference_image, colored_segmentation))
    
    # Save the concatenated image
    cv2.imwrite(output_path, concatenated_image)

def process_image(input_image_path, txt_file_path, reference_image_path, output_image_path, output_reference_path,fracture_only=False):
    
    # Create semantic segmentation map
    segmentation_map = create_semantic_segmentation(input_image_path, txt_file_path,fracture_only=fracture_only)
    
    # Save the original single-channel segmentation map
    Image.fromarray(segmentation_map).save(output_image_path)
    
    # Create colored segmentation for reference image
    colored_segmentation = create_colored_segmentation(segmentation_map)
    
    # Create and save the side-by-side comparison image
    save_image_with_reference(colored_segmentation, reference_image_path, output_reference_path)
    
    print(f"Processed: {os.path.basename(input_image_path)}")
    # from IPython import embed; embed()
    # print(f"  Segmentation map saved as: {output_image_path}")
    # print(f"  Comparison image saved as: {output_reference_path}")
    # print(f"  Unique values in segmentation map: {np.unique(segmentation_map)}")
    # print()

def main():
    parser = argparse.ArgumentParser(description="Batch process semantic segmentation images.")
    parser.add_argument("txt_folder", help="Folder containing .txt files")
    parser.add_argument("reference_folder", help="Folder containing reference images")
    parser.add_argument("input_folder", help="Folder containing input images")
    parser.add_argument("output_folder", help="Folder to save output segmentation images")
    parser.add_argument("output_reference_folder", help="Folder to save output reference images")
    parser.add_argument("--fracture_only", action="store_true", help="only fractures")
    
    args = parser.parse_args()
    
    # Create output folders if they don't exist
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_reference_folder, exist_ok=True)
    
    # Process all input images
    # for input_image_path in glob.glob(os.path.join(args.input_folder, "*.png")):
    txt_files=glob.glob(os.path.join(args.txt_folder, "*.txt"))
    txt_files=sorted(txt_files)
    for txt_file_path in txt_files:
        filename = os.path.basename(txt_file_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        
        # txt_file_path = os.path.join(args.txt_folder, f"{name_without_ext}.txt")
        imname=f"{name_without_ext}.png"
        input_image_path = os.path.join(args.input_folder, imname)
        reference_image_path = os.path.join(args.reference_folder, imname)
        output_image_path = os.path.join(args.output_folder, imname)
        output_reference_path = os.path.join(args.output_reference_folder, imname)
        
        if os.path.exists(input_image_path) and os.path.exists(reference_image_path):
            if not os.path.exists(output_image_path):
                process_image(input_image_path, txt_file_path, reference_image_path, output_image_path, output_reference_path,args.fracture_only)
            else:
                print(f"Skipping {filename}: exists already")
                
        else:
            print(f"Skipping {filename}: Missing txt file or reference image")

if __name__ == "__main__":
    main()