import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import re
import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
import time
from collections import defaultdict

from unixcoder import UniXcoder

# Initialize NLP model and sentence transformer globally
nlp = English()
nlp.add_pipe("sentencizer")  # Add sentence segmentation capability
embedding_model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code=True, device="cuda")

# Set up UniXcoder for code embeddings
device = torch.device("cuda")
code_embedding_model = UniXcoder("microsoft/unixcoder-base")
code_embedding_model.to(device)

def get_single_code_embedding(text: str) -> list:
    """
    Extract embeddings from a code snippet or a natural language query.
    """
    # print(type(text), text)
    tokens_ids = code_embedding_model.tokenize([text],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, nl_embedding = code_embedding_model(source_ids)
    norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)
    norm_nl_embedding = norm_nl_embedding.detach().cpu().numpy()[0]
    return norm_nl_embedding

def text_formatter(text: str) -> str:
    """Cleans and formats text: removes extra newlines and trims whitespace."""
    return text.replace("\n", " ").strip()

def is_code_snippet(text, font):
    """
    A simple function to detect code snippets based on indentation,
    common keywords, and short lines (which may indicate pseudocode).
    """
    code_keywords = ['/', '>', '{', '}', '#', 'void', 'str', 'while', 'if', 'return', 'def', 'accept', 'delete', 'class', 'int', 'float', 'bool', 'end', '=']

    if text.startswith('    '):
        return True
    first_word = text.split()[0] if text.strip() else ""
    if first_word in code_keywords:
        return True
    if text.endswith(';'):
        return True
    if "courier" in font.lower() or "mono" in font.lower():
        return True
    """
    Removing condition for checking line length since it removes shorter sentences.
    if len(text) < 30:
        return True
    """
    return False

def open_and_read_pdf(pdf_path: str, start_page: int, end_page: int, header_height: int, footer_height: int) -> tuple:
    """Opens PDF document and extracts text and code segments per page."""
    doc = fitz.open(pdf_path)  # Open the PDF document
    text_per_page = defaultdict(list)
    code_snippets = dict()

    # Iterate over the pages within the specified range
    for page_num in tqdm(range(start_page, end_page + 1)):
        page = doc.load_page(page_num)  # Load the specific page
        page_height = page.rect.height
        current_code_snippet = []

        # Extract text in block structure
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block: continue # extracted data could be image, and hence may not contain lines
            normal_text = ''

            for line in block["lines"]:
                for span in line["spans"]:
                    bbox = span["bbox"]
                    top_y = bbox[1]  # The top y-coordinate of the span
                    bottom_y = bbox[3]  # The bottom y-coordinate of the span
                    
                    # Exclude text that falls within the header or footer region
                    if top_y > header_height and bottom_y < (page_height - footer_height):
                        text = text_formatter(span["text"])  # Clean and format the text
                        font = span["font"]  # Extract the font name
                        
                        # Check if the text is a code snippet based on font or content
                        if is_code_snippet(text, font):
                            if normal_text:
                                normal_text = ''
                            current_code_snippet.append(text)
                        else:
                            normal_text += ' ' + text
                            
            if normal_text:
                text_per_page[page_num].append(normal_text.strip())
        
        code_snippets[page_num] = ('\n'.join(current_code_snippet))

    return code_snippets, text_per_page

def text_and_code_to_dataframe(text_per_page: dict[list], code_snippets: dict) -> tuple:
    """
    Takes individual chunks from each page and creates a dictionary with relevant statistics
    """
    pages_and_texts = []
    pages_and_code = []

    for page_num in text_per_page:
        text = ''.join(text_per_page[page_num])
        pages_and_texts.append({
                    "page_number": page_num + 1,  # Adjust page number to 1-based indexing
                    "page_char_count": len(text),  # Record character count for the page
                    "page_word_count": len(text.split(" ")), # Record word count
                    "page_sentence_count_raw": len(text.split(". ")), # Raw period-splitted sentence count
                    "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                    "text": text  # Store the extracted text
                })
        
        # add code snippet for corresponding page
        if page_num in code_snippets:
            pages_and_code.append({
                "page_number": page_num + 1,
                "code": code_snippets[page_num] # storing the code snippet without formatting, a separate experiment could embed the code with formatting and check results
            })

    return pages_and_texts, pages_and_code

def sentence_chunking(pages_and_texts: list[dict]) -> None:
    """
    Splits text from each page into individual sentences using spaCy's NLP model.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries containing page text.
    """
    for item in pages_and_texts:
        doc = nlp(item["text"])  # Apply NLP pipeline to extract sentences
        item["sentences"] = [str(sent) for sent in doc.sents]  # Convert sentences to strings

def split_list(input_list: list, chunk_size: int, overlap: int = 2) -> list[list[str]]:
    """
    Splits a list into smaller sublists of a given size.
    Parameters:
        input_list (list): List to be split.
        chunk_size (int): Number of items in each chunk.
        overlap (int): Number of items to overlap between consecutive chunks. 
    Returns:
        list[list]: List of sublists, each with up to chunk_size elements.
    """
    chunks = []
    for i in range(0, len(input_list), chunk_size - overlap):
        chunks.append(input_list[i:i + chunk_size])
        if i + chunk_size >= len(input_list):
            break  # Stop if there are not enough elements for another chunk
    return chunks

def merge_and_filter_chunks(pages_and_texts: list[dict], num_sentence_chunk_size: int, min_token_length: int) -> list[dict]:
    """
    Merges sentences into chunks and filters chunks based on token count.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries containing page sentences.
        num_sentence_chunk_size (int): Number of sentences per chunk.
        min_token_length (int): Minimum token count for chunks to be considered valid.
    
    Returns:
        list[dict]: A list of filtered sentence chunks with metadata.
    """
    pages_and_chunks = []

    for item in pages_and_texts:
        sentence_chunks = split_list(item["sentences"], num_sentence_chunk_size, overlap=2)  # Split sentences into chunks with overlap
        
        for chunk in sentence_chunks:
            # Join sentences into a single string and clean up spacing
            chunk_text = "".join(chunk).replace("  ", " ").strip()
            chunk_text = re.sub(r'\.([A-Z])', r'. \1', chunk_text)  # Ensure correct spacing after periods
            
            chunk_info = {
                "page_number": item["page_number"],
                "sentence_chunk": chunk_text,
                "chunk_char_count": len(chunk_text),  # Record character count of the chunk
                "chunk_word_count": len([word for word in chunk_text.split(" ")]),
                "chunk_token_count": len(chunk_text) / 4  # Approximate 1 token as 4 characters
            }

            if chunk_info["chunk_token_count"] > min_token_length:  # Filter out chunks that are too small
                pages_and_chunks.append(chunk_info)

    return pages_and_chunks

def embed_chunks(pages_and_chunks: list[dict]) -> None:
    """
    Generates embeddings for each text chunk using the pre-loaded SentenceTransformer model.
    
    Parameters:
        pages_and_chunks (list[dict]): List of text chunks for embedding.
    """
    start_time = time.time()

    for item in tqdm(pages_and_chunks):
        # Generate and store the embedding for each chunk of text
        item["text_embeddings"] = embedding_model.encode(item["sentence_chunk"])

    end_time = time.time()
    
    print(f"[INFO] Time taken to generate document embeddings: {end_time-start_time:.5f} seconds.")

def create_code_embeddings(pages_and_code: list[dict]) -> None:
    """
    Generates embeddings for each code snippet.
    Parameters:
        pages_and_chunks (list[dict]): List of text chunks for embedding.
        code_snippets (dict): List of code snippets read page-wise.
    """
    start_time = time.time()

    for item in tqdm(pages_and_code):
        item['code_embeddings'] = get_single_code_embedding(item['code'])

    end_time = time.time()
    print(f"[INFO] Time taken to generate code embeddings: {end_time-start_time:.5f} seconds.")

def process_pdf_for_embeddings(file_path_name: str, start_page: int, end_page: int, num_sentence_chunk_size: int, min_token_length: int, header_height: int=50, footer_height: int=60) -> None:
    """
    Automates the process of extracting text from a PDF, chunking sentences, generating embeddings, and saving results to a CSV.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        start_page (int): Start page number (0-indexed).
        end_page (int): End page number (0-indexed).
        num_sentence_chunk_size (int): Number of sentences per chunk for embedding.
        min_token_length (int): Minimum token count for valid chunks.
        output_file (str): Path to the CSV file for saving results.
    """
    pdf_path = "data/" + file_path_name + ".pdf"
    output_path = file_path_name + "_embeddings.csv"
    code_snippets, text_per_page = open_and_read_pdf(pdf_path, start_page, end_page, header_height, footer_height) # Extract text from PDF
    pages_and_texts, pages_and_code = text_and_code_to_dataframe(text_per_page, code_snippets)  # Create dataframe for text

    sentence_chunking(pages_and_texts)  # Split text into sentences
    pages_and_chunks = merge_and_filter_chunks(pages_and_texts, num_sentence_chunk_size, min_token_length)  # Create and filter chunks
    embed_chunks(pages_and_chunks)  # Generate text embeddings for each chunk

    # Save the final text embeddings to a CSV file
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv("embeddings/text/" + str(output_path), index=False)
    print(f"Embeddings saved to embeddings/text/{output_path}")

    create_code_embeddings(pages_and_code) # Create code database

    code_df = pd.DataFrame(pages_and_code)
    code_df.to_csv("embeddings/code/" + str(output_path), index=False)
    print(f"Embeddings saved to embeddings/code/{output_path}")

# Statistical Analysis 
def calculate_page_statistics(pages_and_texts: list[dict], code_snippets: list[dict]) -> pd.DataFrame:
    """
    Converts the list of page dictionaries to a DataFrame and calculates
    average and minimum token and sentence counts across all pages.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries containing page metadata and text.
    
    Returns:
        pd.DataFrame: A DataFrame with the token and sentence statistics.
    """
    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(pages_and_texts)
    code_df = pd.DataFrame(code_snippets)
    
    # Calculate average and minimum token count
    avg_token_count = df["page_token_count"].mean()
    min_token_count = df["page_token_count"].min()

    # Calculate average and minimum sentence count
    avg_sentence_count = df["page_sentence_count_raw"].mean()
    min_sentence_count = df["page_sentence_count_raw"].min()

    # Print text and code in a single page
    print(df[0]["text"])
    print(df[0]["code"])

    # Print statistics
    print(f"Average Token Count per Page: {avg_token_count:.2f}")
    print(f"Minimum Token Count per Page: {min_token_count}")
    print(f"Average Sentence Count per Page: {avg_sentence_count:.2f}")
    print(f"Minimum Sentence Count per Page: {min_sentence_count}")

    return df


### Testing Code

# code_snippets, pages_and_texts = open_and_read_pdf("data/algorithm-design-manual.pdf", start_page=120, end_page=124, header_height=60, footer_height=50)
# df = calculate_page_statistics(pages_and_texts)
