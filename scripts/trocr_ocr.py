from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

def perform_ocr(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def ocr_on_lines(line_images):
    processor, model = load_model()
    recognized_texts = []
    for line_image in line_images:
        text = perform_ocr(line_image, processor, model)
        recognized_texts.append(text)
    return recognized_texts

if __name__ == "__main__":
    import sys
    from glob import glob

    lines_dir = sys.argv[1]
    line_images = sorted(glob(os.path.join(lines_dir, '*.png')))
    
    recognized_texts = ocr_on_lines(line_images)
    for i, text in enumerate(recognized_texts):
        print(f"Line {i+1}: {text}")
