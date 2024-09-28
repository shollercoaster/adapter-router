from pdf_processing_and_embedding import process_pdf_for_embeddings


# Embeddings for OSTEP
process_pdf_for_embeddings(
    file_path_name="operating_systems_three_easy_pieces", 
    start_page=38, 
    end_page=623, 
    num_sentence_chunk_size=8, 
    min_token_length=30, 
    header_height=70,
    footer_height=100,
)

# Embeddings for Algorithm Design Manual
process_pdf_for_embeddings(
    file_path_name="algorithm-design-manual", 
    start_page=14, 
    end_page=665,
    num_sentence_chunk_size=8, 
    min_token_length=40,
)

# Embeddings for Python Tutorial
process_pdf_for_embeddings(
    file_path_name="python-tutorial", 
    start_page=8, 
    end_page=114,
    num_sentence_chunk_size=4, 
    min_token_length=30,
)

# Embeddings for Neural Network Foundations
process_pdf_for_embeddings(
    file_path_name="neural-network-learning-theoretical-foundations", 
    start_page=15, 
    end_page=370,
    num_sentence_chunk_size=8, 
    min_token_length=30, 
)

# Embeddings for Cognitive Science Foundations 
process_pdf_for_embeddings(
    file_path_name="mind-body-world-cog-sci", 
    start_page=16, 
    end_page=440,
    num_sentence_chunk_size=8, 
    min_token_length=40,
)

