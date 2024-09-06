from pdf_processing_and_embedding import process_pdf_for_embeddings


# Embeddings for OSTEP
process_pdf_for_embeddings(
    pdf_path="data/operating_systems_three_easy_pieces.pdf", 
    start_page=38, 
    end_page=623, 
    num_sentence_chunk_size=8, 
    min_token_length=30, 
    output_file="embeddings/ostep_text_chunks_and_embeddings.csv"
)

# Embeddings for Neural Network Foundations
process_pdf_for_embeddings(
    pdf_path="data/neural-network-learning-theoretical-foundations.pdf", 
    start_page=15, 
    end_page=370,
    num_sentence_chunk_size=8, 
    min_token_length=30, 
    output_file="embeddings/neural_network_text_chunks_and_embeddings.csv"
)


# Embeddings for Neural Network Foundations
process_pdf_for_embeddings(
    pdf_path="data/algorithm-design-manual.pdf", 
    start_page=14, 
    end_page=665,
    num_sentence_chunk_size=10, 
    min_token_length=40,
    output_file="embeddings/algorithm_design_manual_text_chunks_and_embeddings.csv"
)
