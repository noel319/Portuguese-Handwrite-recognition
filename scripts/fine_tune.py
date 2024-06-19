from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from PIL import Image

# Load dataset
dataset = load_from_disk("portuguese_handwritten_dataset")

# Load processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def preprocess_data(examples):
    images = [Image.open(image_path).convert("RGB") for image_path in examples['image']]
    pixel_values = processor(images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(examples['text'], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return {"pixel_values": pixel_values, "labels": labels}

# Preprocess dataset
dataset = dataset.map(preprocess_data, batched=True, remove_columns=["image", "text"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-finetuned-portuguese",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=500,
    learning_rate=5e-5,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
)

# Train
trainer.train()

# Save the model
model.save_pretrained("./trocr-finetuned-portuguese")
processor.save_pretrained("./trocr-finetuned-portuguese")