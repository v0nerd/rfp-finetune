# RFP Fine-Tuning

This project involves fine-tuning two models: BART for summarization and BERT for compliance checking. The raw data is stored in an AWS S3 bucket, preprocessed, and used for training the models.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Set up AWS credentials for S3 access:
Use aws configure to set your credentials.

## Training

1. Run the summarization model training:
```bash
python models/summarization_model/train.py
```

2. Run the compliance model training:
```bash
python models/compliance_model/train.py
```