import torch
from unixcoder import UniXcoder

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
    code_keywords = ['/', '>', '{', '}', '#', 'void', 'str', 'while', 'if', 'return', 'def', 'accept', 'delete', 'int', 'float', 'bool', 'end']

    if text.startswith('    '):
        return True
    first_word = text.split()[0] if text.strip() else ""
    if first_word in code_keywords:
        return True
    """
    if text.endswith(';'):
        return True
    """
    if "courier" in font.lower() or "mono" in font.lower():
        return True
    """
    Removing condition for checking line length since it removes shorter sentences.
    if len(text) < 30:
        return True
    """
    return False

def format_code_snippet(code_snippet: str) -> str:
    """
    Format the code snippet by breaking lines at certain symbols and adding indentation to improve readability.
    Parameters:
        code_snippet (str): The raw code snippet as a single line.

    Returns:
        str: Formatted code snippet with proper line breaks and indentation.
    """
    # Define symbols where we break the line
    break_symbols = ['{', '}', ';']

    # Initialize variables for formatted code and indentation level
    formatted_code = ""
    indent_level = 0
    indent_spaces = 4  # Number of spaces for each indent level

    # Split the code snippet into tokens based on break symbols
    tokens = []
    current_token = ""

    for char in code_snippet:
        current_token += char
        if char in break_symbols:
            tokens.append(current_token.strip())
            current_token = ""

    # Append any remaining characters as the final token
    if current_token.strip():
        tokens.append(current_token.strip())

    # Process each token and apply indentation
    for token in tokens:
        stripped_token = token.strip()

        # Dedent if the token starts with '}', since this ends a block
        if stripped_token.startswith("}"):
            indent_level -= 1

        # Add the token with proper indentation
        formatted_code += " " * (indent_level * indent_spaces) + stripped_token + "\n"

        # Indent if the token ends with '{', since this starts a block
        if stripped_token.endswith("{"):
            indent_level += 1

    return formatted_code.strip()
