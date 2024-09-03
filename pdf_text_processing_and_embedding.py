import random
import pandas as pd
from spacy.lang.en import English # see https://spacy.io/usage for install instructions
import re
from sentence_transformers import SentenceTransformer
import os
import requests
import fitz

from tqdm.auto import tqdm

# Get PDF document
ostep_pdf_path = "data/operating_systems_three_easy_pieces.pdf"

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
def open_and_read_pdf(pdf_path: str, start_page: int, end_page: int) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read,
        start_page (int): Page number marking beginning of chapters in the textbook,
        end_page (int): Page number marking end of chapters in the textbook.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc[start_page:end_page])):  # iterate the document pages from start to end index depending on where pdf begins
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number + 1,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                                "text": text})
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path=ostep_pdf_path, start_page=38, end_page=623)

def sentence_chunking_using_spacy(pages_and_texts: list[dict]) -> None:
    nlp = English()

    # Add a sentencizer pipeline
    nlp.add_pipe("sentencizer")

    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        
        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        
        # Count the sentences 
        item["page_sentence_count_spacy"] = len(item["sentences"])

sentence_chunking_using_spacy(pages_and_texts)

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def merge_and_filter_chunks(num_sentence_chunk_size: int=8, min_token_length: int=30) -> list[dict]:
    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                            slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    # Split each chunk into its own item
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            
            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
            
            pages_and_chunks.append(chunk_dict)

    # Filtering chunks smaller than min_token_length
    df = pd.DataFrame(pages_and_chunks)

    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    pages_and_chunks_over_min_token_len[:2]
    return pages_and_chunks_over_min_token_len

def embedding_text_chunks(pages_and_chunks_over_min_token_len: list[dict]) -> None:
    embedding_model = SentenceTransformer(model_name_or_path="nvidia/NV-Embed-v2", 
                                      trust_remote_code=True,
                                      device="cuda")

    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Save embeddings to file
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    embeddings_df_save_path = "ostep_text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

pages_and_chunks_over_min_token_len = merge_and_filter_chunks(num_sentence_chunk_size=8, min_token_length=30)
embedding_text_chunks(pages_and_chunks_over_min_token_len)
