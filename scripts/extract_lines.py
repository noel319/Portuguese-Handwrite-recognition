import cv2
from PIL import Image
import os

def extract_lines(image, contours, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    line_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        line_image = image[y:y+h, x:x+w]
        line_image_pil = Image.fromarray(line_image)
        line_image_path = os.path.join(output_dir, f'line_{i+1}.png')
        line_image_pil.save(line_image_path)
        line_images.append(line_image_path)
    return line_images

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    thresh_image_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresh_image = cv2.imread(thresh_image_path, cv2.IMREAD_GRAYSCALE)
    
    from detect_lines import detect_lines
    contours = detect_lines(thresh_image)
    
    line_images = extract_lines(image, contours, output_dir)
    print(f"Extracted {len(line_images)} lines.")
