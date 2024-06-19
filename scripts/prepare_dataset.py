import os
from datasets import Dataset, DatasetDict
from PIL import Image

def load_images_and_labels(data_dir):
    images=[]
    texts=[]
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                text_path = image_path.replace('.jpg','.txt').replace('.png', '.txt')
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    images.append(image_path)
                    texts.append(text)
    return images, texts

def prepare_dataset(data_dir):
    images, texts = load_images_and_labels(data_dir)
    dataset = Dataset.from_dict({"image":images, "text":texts})
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


if __name__ == "__main__":
    data_dir = "data/"
    dataset = prepare_dataset(data_dir)
    dataset.save_to_disk("portuguese_handwritten_dataset")