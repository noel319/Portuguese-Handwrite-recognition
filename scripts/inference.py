from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the fine-tuned model
processor = TrOCRProcessor.from_pretrained('./trocr-finetuned-portuguese')
model = VisionEncoderDecoderModel.from_pretrained('./trocr-finetuned-portuguese')

def perform_ocr(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    text = perform_ocr(image_path)
    print("Recognized text:", text)
