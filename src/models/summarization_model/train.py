from transformers import Trainer, TrainingArguments, BartForConditionalGeneration
from datasets import Dataset
from peft import get_peft_model, LoraConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.summarization_model.model import SummarizationModel
import os


def apply_lora_to_bart(model_name: str, lora_config: dict):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # Apply LoRA to the model
    model = get_peft_model(model, LoraConfig(**lora_config))
    return model


# Function to tokenize the data
# Load the processed CSV
df = pd.read_csv("datasets/processed/processed.csv")

# Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)  # 90% train, 10% validation


# Tokenization function
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["content"], truncation=True, padding="max_length", max_length=1024
    )


# Prepare the dataset for fine-tuning (from processed.csv)
def prepare_dataset(df, tokenizer):
    # Tokenize the 'content' for input and 'summary' for target
    inputs = tokenizer(
        df["content"].tolist(), truncation=True, padding="max_length", max_length=1024
    )
    targets = tokenizer(
        df["summary"].tolist(), truncation=True, padding="max_length", max_length=1024
    )

    # Add labels to the tokenized dataset (labels are the target summaries)
    inputs["labels"] = targets["input_ids"]

    # Convert the tokenized data into a Hugging Face Dataset object
    dataset = Dataset.from_dict(inputs)

    return dataset


def train_summarization_model():
    model = SummarizationModel()

    model.download_from_s3("rfp-models", "summarization_model_fine_tuned")

    model.model = apply_lora_to_bart(model.model_name, model.config["lora"])

    for param in model.model.parameters():
        param.requires_grad = True

    ## Prepare the datasets
    train_dataset = prepare_dataset(train_df, model.tokenizer)
    test_dataset = prepare_dataset(val_df, model.tokenizer)

    training_args = TrainingArguments(
        output_dir=model.config["model"]["model_path"],
        evaluation_strategy="epoch",
        learning_rate=model.config["training"]["learning_rate"],
        per_device_train_batch_size=model.config["training"]["batch_size"],
        num_train_epochs=model.config["training"]["epochs"],
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Provide validation dataset for evaluation
        tokenizer=model.tokenizer,
    )

    trainer.train()

    # After training, upload the updated model to S3
    # model.upload_to_s3(os.getenv("S3_BUCKET_NAME"), "summarization_model_path")
    model.upload_to_s3("rfp-models", "summarization_model_fine_tuned")
    print("Model training complete and uploaded to S3.")


if __name__ == "__main__":
    train_summarization_model()
