from pdf_processing_and_embedding import process_pdf_for_embeddings


# Embeddings for OSTEP
process_pdf_for_embeddings(
    pdf_path="operating_systems_three_easy_pieces", 
    start_page=38, 
    end_page=623, 
    num_sentence_chunk_size=8, 
    min_token_length=30, 
    header_height=70,
    footer_height=100,
)

# Embeddings for Algorithm Design Manual
process_pdf_for_embeddings(
    pdf_path="algorithm-design-manual.pdf", 
    start_page=14, 
    end_page=665,
    num_sentence_chunk_size=8, 
    min_token_length=40,
)
'''
# Embeddings for Neural Network Foundations
process_pdf_for_embeddings(
    pdf_path="data/neural-network-learning-theoretical-foundations.pdf", 
    start_page=15, 
    end_page=370,
    num_sentence_chunk_size=8, 
    min_token_length=30, 
    output_path="neural_network_text_chunks_and_embeddings.csv"
)

# Embeddings for Cognitive Science Foundations 
process_pdf_for_embeddings(
    pdf_path="data/mind-body-world-cog-sci.pdf", 
    start_page=16, 
    end_page=440,
    num_sentence_chunk_size=8, 
    min_token_length=40,
    output_path="cog_sci_foundations_text_chunks_and_embeddings.csv"
)
'''
