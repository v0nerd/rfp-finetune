import boto3
import os


def download_file_from_s3(bucket_name, s3_key, local_path):
    """
    Downloads a single file from S3 to the local path.
    """
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Downloaded: {local_path}")
    except Exception as e:
        print(f"Error downloading {s3_key}: {str(e)}")


def load_raw_files_from_s3(bucket_name, raw_folder):
    """
    Loads and downloads all files from the given S3 raw folder sequentially.
    """
    s3 = boto3.client("s3")
    raw_files = []

    # List all files in the raw folder (handle pagination if there are many files)
    continuation_token = None
    while True:
        list_args = {"Bucket": bucket_name, "Prefix": raw_folder}
        if continuation_token:
            list_args["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**list_args)

        # Handle paginated response
        continuation_token = response.get("NextContinuationToken")

        # Process each file in the current response
        for obj in response.get("Contents", []):
            file_key = obj["Key"]
            local_file_path = os.path.join("datasets/raw", os.path.basename(file_key))
            raw_files.append((bucket_name, file_key, local_file_path))

        # If there are no more files, break the loop
        if not continuation_token:
            break

    # Download files one by one sequentially
    for bucket_name, file_key, local_file_path in raw_files:
        download_file_from_s3(bucket_name, file_key, local_file_path)

    return raw_files


if __name__ == "__main__":
    bucket_name = os.getenv(
        "S3_BUCKET_NAME"
    )  # Ensure you have the S3 bucket name in the environment variables
    raw_folder = "raw/"

    files = load_raw_files_from_s3("von-rfp-datasets", raw_folder)
    print(f"Loaded {len(files)} raw files from S3.")
