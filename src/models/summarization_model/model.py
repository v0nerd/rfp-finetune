import yaml

from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import os
import boto3
from botocore.exceptions import NoCredentialsError


config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

# Load configuration
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


class SummarizationModel:
    def __init__(self, model_name=config["model"]["model_name"], config=config):
        self.config = config
        if os.path.isdir(config["model"]["model_path"]):
            self.model_name = config["model"]["model_path"]
        else:
            self.model_name = model_name
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)

    def train(self, train_dataset):
        # Implement your training loop here
        pass

    def generate_summary(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def upload_to_s3(self, bucket_name, model_key):
        # Save the model and tokenizer locally
        self.model.save_pretrained(config["model"]["model_path"])
        self.tokenizer.save_pretrained(config["model"]["model_path"])

        # Create an S3 client using boto3
        s3_client = boto3.client("s3")

        # Upload the model files recursively
        def upload_directory_to_s3(local_path, s3_path):
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(
                        local_file_path, local_path
                    )  # Relative path to the model directory
                    s3_file_path = os.path.join(s3_path, relative_path)

                    try:
                        # Upload the file to S3
                        print(
                            f"Uploading {local_file_path} to s3://{bucket_name}/{s3_file_path}"
                        )
                        s3_client.upload_file(
                            local_file_path, bucket_name, s3_file_path
                        )
                    except NoCredentialsError:
                        print("Credentials not available")
                    except Exception as e:
                        print(f"Error uploading {local_file_path}: {str(e)}")

        # Assuming the model directory is the `model_key` directory
        upload_directory_to_s3(config["model"]["model_path"], model_key)

        print(f"Model successfully uploaded to s3://{bucket_name}/{model_key}")

    def download_from_s3(self, bucket_name, model_key):
        # Create an S3 client
        s3_client = boto3.client("s3")

        # Download the model files recursively from S3
        def download_directory_from_s3(s3_bucket, s3_prefix, local_path):
            if not os.path.exists(local_path):
                os.makedirs(local_path)

            # List objects under the specified S3 prefix (model_key)
            try:
                response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
                if "Contents" in response:
                    for obj in response["Contents"]:
                        s3_file_key = obj["Key"]
                        local_file_path = os.path.join(
                            local_path, os.path.relpath(s3_file_key, s3_prefix)
                        )

                        # Create directories if they do not exist
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                        # Download the file from S3
                        print(f"Downloading {s3_file_key} to {local_file_path}")
                        s3_client.download_file(s3_bucket, s3_file_key, local_file_path)
                else:
                    print(f"No files found for {s3_prefix} in bucket {s3_bucket}")
            except NoCredentialsError:
                print("Credentials not available")
            except Exception as e:
                print(f"Error downloading from S3: {str(e)}")

        # Download the model and tokenizer from the S3 bucket to the local model_key directory
        download_directory_from_s3(
            bucket_name, model_key, config["model"]["model_path"]
        )
