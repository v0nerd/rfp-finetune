import pandas as pd
import os

from .utils import (
    extract_text_from_docx,
    extract_text_from_pdf,
    clean_text,
    get_section_from_file,
)


def preprocess_raw_files(file_paths):
    data = []
    for file_path in file_paths:
        if file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        content = clean_text(text)
        
        # Extract the content from the file based on the type
        file_extension = filename.split(".")[-1].lower()
        
        raw_section_text = get_section_from_file([file_path], file_extension)

        section_text = clean_text(raw_section_text)

        # Simple example to create mock data (content, summary, section_text, compliance)
        # Adjust according to your actual requirements
        data.append(
            {
                "title": os.path.basename(file_path),
                "content": content,
                "summary": "Generated summary for "
                + os.path.basename(file_path),  # Dummy summary
                "section_text": section_text,
                "compliance": "compliant",  # Dummy compliance label, replace as needed
            }
        )

    return pd.DataFrame(data)


def save_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # raw_files = ["datasets/raw/file1.docx", "datasets/raw/file2.pdf"]  # Example paths
    raw_files = []
    for filename in os.listdir("datasets/raw"):
        raw_files.append(os.path.join("datasets/raw", filename))
    processed_data = preprocess_raw_files(raw_files)
    save_to_csv(processed_data, "datasets/processed/processed.csv")
    print("Preprocessed data saved to datasets/processed/processed.csv.")
