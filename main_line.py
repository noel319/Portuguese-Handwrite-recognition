import os
import sys

def main(input_image_path, output_dir):
    from scripts.preprocess_image import preprocess_image
    from scripts.detect_lines import detect_lines
    from scripts.extract_lines import extract_lines
    
    # Preprocess the image
    image, thresh_image = preprocess_image(input_image_path)
    preprocessed_image_path = "preprocessed_image.jpg"
    cv2.imwrite(preprocessed_image_path, thresh_image)
    
    # Detect lines
    contours = detect_lines(thresh_image)
    
    # Extract and save lines
    line_images = extract_lines(image, contours, output_dir)
    
    print(f"Extracted lines saved in: {output_dir}")
    for line_image in line_images:
        print(line_image)

if __name__ == "__main__":
    input_image_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    main(input_image_path, output_dir)
