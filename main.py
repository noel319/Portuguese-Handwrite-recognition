import argparse
from scripts import prepare_dataset, fine_tune, inference

def main():
    parser = argparse.ArgumentParser(description="Portuguese Handwritten Text Recognition with TrOCR")
    parser.add_argument("--prepare-dataset", action="store_true", help="Prepare the dataset")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model")
    parser.add_argument("--inference", type=str, help="Perform OCR on a given image")
    
    args = parser.parse_args()

    if args.prepare_dataset:
        prepare_dataset.prepare_dataset("data/")
    elif args.fine_tune:
        fine_tune.fine_tune_model()
    elif args.inference:
        text = inference.perform_ocr(args.inference)
        print("Recognized text:", text)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
