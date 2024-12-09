import re

from docx import Document
from PyPDF2 import PdfReader

from dotenv import load_dotenv

load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)


def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary whitespace and newlines.
    """
    # Remove page number headers/footers like "Page X of Y"
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"Page \d+-\d+", "", text)
    text = re.sub(r"Page \d+", "", text)

    # 1. Remove lines that appear to be tables (based on multiple words separated by spaces or tabs)
    # Match lines with multiple spaces or tabs (suggesting columns in a table)
    # text = re.sub(r"^[^\n]*\s{2,}[^\n]*$", "", text, flags=re.MULTILINE)

    # 2. Remove rows that consist only of numbers or numbers and dashes (commonly in tables)
    # Match rows of digits (potential numeric data rows in tables)
    text = re.sub(r"^\d[\d\s-]*$", "", text, flags=re.MULTILINE)

    # 3. Remove lines containing table separators (like dashes or pipes, commonly used in table headers or dividers)
    # Match lines that consist only of dashes or pipes
    text = re.sub(r"^[\-\|]+$", "", text, flags=re.MULTILINE)

    # 4. Clean up extra spaces or redundant line breaks after table removal
    text = re.sub(r"\n+", "\n", text)  # Normalize line breaks
    text = text.strip()  # Strip leading/trailing whitespace

    # Remove multiple newlines and extra spaces
    # 1. Remove multiple line breaks and normalize to a single space
    text = re.sub(r"\n+", " ", text)  # Replace multiple newlines with a single space

    # 2. Replace tabs and special whitespace characters with a single space
    text = re.sub(r"\t+", " ", text)  # Replace tabs with a single space
    text = re.sub(r"\xa0", " ", text)  # Replace non-breaking spaces with normal spaces

    # 3. Replace multiple consecutive spaces with a single space
    text = re.sub(r" +", " ", text)  # Replace multiple spaces with a single space

    # 4. Strip leading and trailing spaces
    # text = text.strip()  # Remove leading and trailing whitespace

    # Remove special characters (e.g., non-breaking spaces or control characters)
    text = re.sub(r"\xa0", " ", text)  # Replace non-breaking spaces
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters

    # Optionally, you could also remove extra punctuation marks or specific symbols
    # text = re.sub(r'[^\w\s,.!?]', '', text)

    return text


def get_section_from_file(
    file_paths: list = ["datasets/raw/0.pdf"], file_extension: str = "pdf"
) -> str:
    # set up parser
    parser = LlamaParse(result_type="text")  # "markdown" and "text" are available

    file_extractor = {".pdf": parser}

    if file_extension == "docx":
        # use SimpleDirectoryReader to parse our file
        file_extractor = {".docx": parser}
    documents = SimpleDirectoryReader(
        input_files=file_paths, file_extractor=file_extractor
    ).load_data()

    # create an index from the parsed markdown
    index = VectorStoreIndex.from_documents(documents)

    # create a query engine for the index
    query_engine = index.as_query_engine()

    # query the engine
    query = """
    Classify each section of attached RFP document according to the categories provided as following. If section is provided, give detailed information from the document:

    Categories:
    1. Introduction
        1.1 Purpose of the RFP
        1.2 Background
        1.3 Scope of the RFP
    2. Objectives
        2.1 Project Goals
        2.2 Expected Outcomes
    3. Scope of Work
        3.1 Deliverables.
            Design, development, and deployment of <product/service>.
            User training and documentation.
            Ongoing maintenance and support for <X> months/years.
        3.2 Timeline
        3.3 Project Requirements
    4. Eligibility Criteria
        4.1 Vendor Qualifications
        4.2 Financial Stability
        4.3 References
            Project description.
            Scope of work.
            Project outcomes.
    5. Proposal Instructions
        5.1 Proposal Submission
        5.2 Proposal Format
        5.3 Questions and Clarifications
    6. Evaluation Criteria
        6.1 Evaluation Process
        6.2 Scoring System
    7. Budget and Pricing
        7.1 Pricing Structure
        7.2 Payment Terms
        7.3 Budget Limitations
    8. Timeline
        8.1 Project Milestones
        8.2 Deliverable Deadlines
    9. Terms and Conditions
        9.1 Confidentiality
        9.2 Indemnification
        9.3 Governing Law
        9.4 Contractual Agreement
    10. Contact Information
        <Name>
        <Title>
        <Company Name>
        <Phone Number>
        <Email Address>
    """
    response = query_engine.query(query).response

    return response
