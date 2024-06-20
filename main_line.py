import os
import sys
import cv2

def main(input_image_path, output_dir):
    from scripts.preprocess_image import preprocess_image
    from scripts.detect_lines import detect_lines
    from scripts.extract_lines import extract_lines
    from scripts.trocr_ocr import ocr_on_lines

    # Preprocess the image
    image, thresh_image = preprocess_image(input_image_path)
    preprocessed_image_path = os.path.join(output_dir, "preprocessed_image.jpg")
    cv2.imwrite(preprocessed_image_path, thresh_image)
    
    # Detect lines
    contours = detect_lines(thresh_image)
    
    # Extract and save lines
    lines_dir = os.path.join(output_dir, "lines")
    line_images = extract_lines(image, contours, lines_dir)
    
    # Perform OCR on lines
    recognized_texts = ocr_on_lines(line_images)
    
    # Save recognized text to a file
    with open(os.path.join(output_dir, "recognized_texts.txt"), "w", encoding="utf-8") as f:
        for i, text in enumerate(recognized_texts):
            f.write(f"Line {i+1}: {text}\n")
    
    print(f"Recognized texts saved in: {os.path.join(output_dir, 'recognized_texts.txt')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <output_dir>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    main(input_image_path, output_dir)
