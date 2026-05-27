"""
Helper Functions for Controllable RAG System

This module contains utility functions for text processing, document manipulation,
PDF handling, similarity analysis, and metric evaluation for RAG applications.
"""

# Standard library imports
import re
import textwrap

# Third-party imports
import tiktoken
import PyPDF2
import pylcs
import pandas as pd
import dill
from langchain_core.documents import Document


# =============================================================================
# TEXT PROCESSING FUNCTIONS
# =============================================================================

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a given string using a specified encoding.

    Args:
        string (str): The input string to tokenize.
        encoding_name (str): The name of the encoding to use (e.g., 'cl100k_base').

    Returns:
        int: The number of tokens in the string according to the specified encoding.
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents (list): A list of document objects, each with a 'page_content' attribute.

    Returns:
        list: The modified list of documents with tab characters replaced by spaces.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents


def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text (str): The input text string.

    Returns:
        str: The text string with double newlines replaced by single newlines.
    """
    cleaned_text = re.sub(r'\n\n', '\n', text)
    return cleaned_text


def escape_quotes(text):
    """
    Escapes both single and double quotes in a string.

    Args:
        text (str): The string to escape.

    Returns:
        str: The string with single and double quotes escaped.
    """
    return text.replace('"', '\\"').replace("'", "\\'")


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int, optional): The width at which to wrap the text. Defaults to 120.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)


# =============================================================================
# PDF PROCESSING FUNCTIONS
# =============================================================================

def split_into_chapters(book_path):
    """
    Splits a PDF book into sections/chapters based on common heading patterns.

    Supports multiple languages and formats:
    - English: CHAPTER ONE, CHAPTER 1, etc.
    - Spanish: CAPÍTULO UNO, CAPÍTULO 1, TÍTULO, SECCIÓN, ARTÍCULO, etc.
    - Numbered sections: 1., 1.1, I., etc.
    - Falls back to splitting by pages if no headings are found.

    Args:
        book_path (str): The path to the PDF book file.

    Returns:
        list: A list of Document objects, each representing a section with its 
              text content and section number metadata.
    """
    with open(book_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        documents = pdf_reader.pages

        # Concatenate text from all pages
        text = " ".join([doc.extract_text() for doc in documents])

        # Try multiple heading patterns in order of specificity.
        # The patterns use lookahead to keep the heading text.
        patterns = [
            # English: CHAPTER ONE, CHAPTER TWO, ...
            r'(CHAPTER\s+[A-Z]+(?:\s+[A-Z]+)*)',
            # English/Spanish: CHAPTER 1, CAPÍTULO 1, Chapter 1, etc.
            r'((?:CHAPTER|CAPÍTULO|Chapter|Capítulo)\s+\d+)',
            # Spanish legal/manual: TÍTULO, CAPÍTULO, SECCIÓN, ARTÍCULO (with optional number)
            r'((?:TÍTULO|CAPÍTULO|SECCIÓN|ARTÍCULO|TITULO|CAPITULO|SECCION|ARTICULO)\s+(?:[IVXLCDM]+|\d+)[.:]?)',
            # Numbered headings: 1., 2., 1.1, etc. (at start of semantic blocks)
            r'(\n\s*\d+(?:\.\d+)*\s+(?:[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑa-záéíóúñ\s]{4,}))',
            # Roman numeral sections: I., II., III., etc.
            r'(\n\s*[IVXLCDM]+\.?\s+(?:[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑa-záéíóúñ\s]{4,}))',
        ]

        chapters = None
        for pattern in patterns:
            candidate = re.split(pattern, text)
            # A successful split should produce at least 3 pieces
            # (leading text, first heading, first body)
            if len(candidate) >= 3:
                chapters = candidate
                break

        # If no patterns matched, fall back to splitting by pages
        if chapters is None:
            chapter_docs = []
            for i, doc in enumerate(documents):
                page_text = doc.extract_text()
                if page_text and page_text.strip():
                    chapter_doc = Document(
                        page_content=page_text,
                        metadata={"chapter": i + 1}
                    )
                    chapter_docs.append(chapter_doc)
            return chapter_docs

        # Create Document objects with chapter metadata
        chapter_docs = []
        chapter_num = 1
        # The split result has a predictable structure:
        # [pre-content, heading1, body1, heading2, body2, ...]
        # We start at index 1 and step by 2 to get (heading, body) pairs.
        for i in range(1, len(chapters) - 1, 2):
            heading = chapters[i]
            body = chapters[i + 1] if (i + 1) < len(chapters) else ""
            chapter_text = heading + " " + body
            if chapter_text.strip():
                doc = Document(page_content=chapter_text, metadata={"chapter": chapter_num})
                chapter_docs.append(doc)
                chapter_num += 1

        # If the split logic produced no documents (corner case),
        # fall back to the page-by-page approach.
        if not chapter_docs:
            for i, doc in enumerate(documents):
                page_text = doc.extract_text()
                if page_text and page_text.strip():
                    chapter_doc = Document(
                        page_content=page_text,
                        metadata={"chapter": i + 1}
                    )
                    chapter_docs.append(chapter_doc)

    return chapter_docs


def extract_book_quotes_as_documents(documents, min_length=50):
    """
    Extracts quotes from documents and returns them as separate Document objects.

    Args:
        documents (list): List of Document objects to extract quotes from.
        min_length (int, optional): Minimum length of quotes to extract. Defaults to 50.

    Returns:
        list: List of Document objects containing extracted quotes.
    """
    quotes_as_documents = []
    # Match straight or curly quoted passages longer than min_length characters.
    quote_pattern_longer_than_min_length = re.compile(rf'["“](.{{{min_length},}}?)["”]', re.DOTALL)

    for doc in documents:
        content = doc.page_content
        content = content.replace('\n', ' ')
        found_quotes = quote_pattern_longer_than_min_length.findall(content)
        
        for quote in found_quotes:
            quote_doc = Document(page_content=quote)
            quotes_as_documents.append(quote_doc)
    
    if quotes_as_documents:
        return quotes_as_documents

    # Some PDF extractions lose quotation marks. Fall back to substantial text
    # passages so downstream quote retrieval still has a non-empty FAISS index.
    for doc in documents:
        content = doc.page_content.replace('\n', ' ').strip()
        if len(content) >= min_length:
            quotes_as_documents.append(Document(page_content=content[:1000]))

    return quotes_as_documents


# =============================================================================
# SIMILARITY AND ANALYSIS FUNCTIONS
# =============================================================================

def is_similarity_ratio_lower_than_th(large_string, short_string, th):
    """
    Checks if the similarity ratio between two strings is lower than a given threshold.

    Uses the Longest Common Subsequence (LCS) algorithm to calculate similarity.

    Args:
        large_string (str): The larger string to compare.
        short_string (str): The shorter string to compare.
        th (float): The similarity threshold (0.0 to 1.0).

    Returns:
        bool: True if the similarity ratio is lower than the threshold, False otherwise.
    """
    # Calculate the length of the longest common subsequence (LCS)
    lcs = pylcs.lcs_sequence_length(large_string, short_string)

    # Calculate the similarity ratio
    similarity_ratio = lcs / len(short_string)

    # Check if the similarity ratio is lower than the threshold
    return similarity_ratio < th


def analyse_metric_results(results_df):
    """
    Analyzes and prints the results of various RAG evaluation metrics.

    Args:
        results_df (pandas.DataFrame): A pandas DataFrame containing the metric results.
    """
    metric_descriptions = {
        "faithfulness": "Measures how well the generated answer is supported by the retrieved documents.",
        "answer_relevancy": "Measures how relevant the generated answer is to the question.",
        "context_precision": "Measures the proportion of retrieved documents that are actually relevant.",
        "context_relevancy": "Measures how relevant the retrieved documents are to the question.",
        "context_recall": "Measures the proportion of relevant documents that are successfully retrieved.",
        "context_entity_recall": "Measures the proportion of relevant entities mentioned in the question that are also found in the retrieved documents.",
        "answer_similarity": "Measures the semantic similarity between the generated answer and the ground truth answer.",
        "answer_correctness": "Measures whether the generated answer is factually correct."
    }

    for metric_name, metric_value in results_df.items():
        print(f"\n**{metric_name.upper()}**")

        # Extract the numerical value from the Series object
        if isinstance(metric_value, pd.Series):
            metric_value = metric_value.values[0]

        # Print explanation and score for each metric
        if metric_name in metric_descriptions:
            print(metric_descriptions[metric_name])
            print(f"Score: {metric_value:.4f}")
        else:
            print(f"Score: {metric_value:.4f}")


# =============================================================================
# OBJECT SERIALIZATION FUNCTIONS
# =============================================================================

def save_object(obj, filename):
    """
    Save a Python object to a file using dill serialization.
    
    Args:
        obj: The Python object to save.
        filename (str): The name of the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        dill.dump(obj, file)
    print(f"Object has been saved to '{filename}'.")


def load_object(filename):
    """
    Load a Python object from a file using dill deserialization.
    
    Args:
        filename (str): The name of the file from which the object will be loaded.
    
    Returns:
        object: The loaded Python object.
    """
    with open(filename, 'rb') as file:
        obj = dill.load(file)
    print(f"Object has been loaded from '{filename}'.")
    return obj


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
# save_object(plan_and_execute_app, 'plan_and_execute_app.pkl')
# plan_and_execute_app = load_object('plan_and_execute_app.pkl')
