from transformers import (
    Trainer,
    TrainingArguments,
    BertForSequenceClassification,
    BertTokenizer,
)
from datasets import Dataset
from src.models.compliance_model.model import ComplianceModel
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def prepare_dataset(tokenizer):
    df = pd.read_csv("datasets/processed/processed.csv")

    # Split the dataset into train and validation (90/10 split)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Convert the pandas DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenize the 'section_text' and add the 'compliance' as labels
    def tokenize_function(examples):
        return tokenizer(
            examples["section_text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Label encoding for compliance labels (if not already encoded)
    label_map = {label: i for i, label in enumerate(df["compliance"].unique())}
    train_dataset = train_dataset.map(lambda x: {"labels": label_map[x["compliance"]]})
    test_dataset = test_dataset.map(lambda x: {"labels": label_map[x["compliance"]]})

    return train_dataset, test_dataset, label_map


def train_compliance_model():
    model = ComplianceModel()
    model.download_from_s3("rfp-models", "compliance_model_fine_tuned")

    # Preprocess and prepare the dataset for the model
    # Assuming the dataset has 'section_text' and 'compliance' columns
    
    for param in model.model.parameters():
        param.requires_grad = True
    
    # Prepare the dataset
    train_dataset, test_dataset, label_map = prepare_dataset(model.tokenizer)

    training_args = TrainingArguments(
        output_dir=model.config["model"]["model_path"],
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
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
    # model.upload_to_s3(os.getenv("S3_BUCKET_NAME"), "compliance_model_path")
    model.upload_to_s3("rfp-models", "compliance_model_fine_tuned")
    print("Model training complete and uploaded to S3.")


if __name__ == "__main__":
    train_compliance_model()
