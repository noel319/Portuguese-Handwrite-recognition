import cv2

def detect_lines(thresh_image):
    # Detect contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by their y position to maintain line order
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    
    return contours

if __name__ == "__main__":
    import sys
    thresh_image_path = sys.argv[1]
    thresh_image = cv2.imread(thresh_image_path, cv2.IMREAD_GRAYSCALE)
    contours = detect_lines(thresh_image)
    print(f"Detected {len(contours)} lines.")
