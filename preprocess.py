#!/usr/bin/env python3
"""
Image Preprocessing Tool
This script applies image enhancement to a single image or a folder of images
and saves the preprocessed results. It is a single, self-contained file.

Usage:
    python single_preprocess_script.py image.jpg
    python single_preprocess_script.py folder_with_images/
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

def enhance_image(image, enhance_images=True):
    """
    Applies image enhancement using contrast stretching and bilateral filtering.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        enhance_images (bool): Whether to apply the enhancement.

    Returns:
        np.ndarray: The enhanced image.
    """
    if not enhance_images:
        return image
    
    try:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast stretching (histogram normalization)
        stretched_image = np.zeros_like(gray_image)
        cv2.normalize(gray_image, stretched_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered_image = cv2.bilateralFilter(stretched_image, d=9, sigmaColor=10, sigmaSpace=100)
        
        # Convert back to BGR for model (YOLOv8 expects color input)
        enhanced_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

        return enhanced_image
    except Exception as e:
        print(f"⚠️ Warning: Image enhancement failed due to an error: {e}")
        return image

def process_image(image_path):
    """
    Loads, enhances, and saves a single image.

    Args:
        image_path (Path): Path to the image file.
    """
    try:
        # Load the image
        print(f"🔧 Preprocessing: {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Error: Could not read image from {image_path}. It might be corrupted or in an unsupported format.")
            return
            
        # Apply the enhancement using the local function
        enhanced_image = enhance_image(image, enhance_images=True)
        
        # Define output path
        output_folder = Path("enhanced_images")
        output_folder.mkdir(exist_ok=True)
        
        image_name = image_path.stem
        output_filename = f"{image_name}{image_path.suffix}"
        output_path = output_folder / output_filename
        
        # Save the enhanced image
        cv2.imwrite(str(output_path), enhanced_image)
        print(f"✅ Saved enhanced image to: {output_path}")

    except Exception as e:
        print(f"⚠️ Failed to process {image_path.name}: {e}")
        
def main():
    """Main function to parse arguments and run the processing."""
    parser = argparse.ArgumentParser(description="Apply image enhancement preprocessing to images.")
    parser.add_argument("input", nargs='?', help="Image file or folder path")
    
    args = parser.parse_args()
    
    if not args.input:
        print("\n📁 Please provide an image file or folder:")
        print("   Example: python single_preprocess_script.py my_image.jpg")
        print("   Example: python single_preprocess_script.py photos_folder/")
        args.input = input("\nEnter path: ").strip().strip('"')
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Error: Path not found: {input_path}")
        sys.exit(1)
        
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            process_image(input_path)
        else:
            print(f"❌ Error: Unsupported file format: {input_path.suffix}")
    
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in {input_path}")
        else:
            print(f"📂 Processing {len(image_files)} images...")
            for image_file in sorted(image_files):
                process_image(image_file)
            print("\n✅ All images processed!")
    
    else:
        print(f"❌ Error: The provided path is not a file or directory: {input_path}")

if __name__ == "__main__":
    main()
