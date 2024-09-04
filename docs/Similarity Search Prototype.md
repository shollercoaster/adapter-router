# Prototype
Created vector embeddings of textbook OSTEP (Operating Systems: Three Easy Pieces) using embedding model and got top 5 passages on finding cosine similarity for a specific query.
# Model choices
## mpnet_base_v2
- doesn't care about dimensions of query and document as long as chunked sentences don't cross model input dimensions (384 tokens). after encoding all embeddings are of size 768.
- can perform both cosine_similarity and dot products (implemented by torch libraries)
- better than top embedding models cause by using inbuilt similarity functions we can also get indices of topk results, allowing us to look at top k passages also. Didn't see similar functionality in other models.
- not in the top rankings of MTEB leaderboard
## [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)
- too big, couldn't load (my cache is full, might be the conda environments)
## [stella_en_1.5B_v5](https://huggingface.co/dunzhang/stella_en_1.5B_v5)
- must do dimensionality analysis to ensure similar dimensions of both query and document embeddings.
- embeddings are numpy arrays, so need to be explicitly converted to pytorch tensors for any calculations
- for some reason the SentenceTransformer inbuilt model.similarity is not a recognised function and does not work
- torch inbuilt dot product methods didn't work initially since both embeddings were of different sizes, then not on cuda device, and then torch.topk showed that selected k was out of range (even when k=1) for the dot product (which was this: `tensor([[820.0814]], device='cuda:0')`)
- the direct embeddings calculated here were also of a fixed dimension: (1024,) but when saved into csv along with the bigger `text_chunked_embedding_df` (which also had per_page token, word and sentence counts, total 6 params) changed dimensionality into (768, 6), which later stayed as a 2d tensor even when just the embeddings df was fetched. Not sure why this happened.
# Code Workflow
## Text Embedding
PDF reading -> text processing (reading data from chapters only, removing pages with less tokens) -> splitting list into sentences -> sentence chunking -> merging and filtering chunks -> vector embeddings of sentence chunks -> saving to csv -> 
## Similarity Search
loading csv -> processing text embedding -> processing query embedding -> dot product / cosine similarity calculations -> generating topk scores -> wrapping text to show top k passages
# Results
- cosine_similarity shows more relevant passages than dot_product
- `num_sentence_chunk_size` gives better results (=8 gave top score of 0.82, =16 gave top score of 0.75)
# Next Steps
- replicate process for 3-4 more software textbooks
- query matching with appropriate textbook embedding 
- think about encodings for adapters
# Caveats
- Text processing looks different for every text corpus (still a textbook here), so building a one-size-fits-all robust system for document processing atleast might not be so straightforward
- Might use a different embedding model for better results