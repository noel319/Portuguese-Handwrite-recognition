import cv2

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve line detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return image, thresh

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    image, thresh = preprocess_image(image_path)
    cv2.imwrite("preprocessed_image.jpg", thresh)
