# Prototype
Created vector embeddings of textbooks OSTEP (Operating Systems: Three Easy Pieces), Neural Networks Theory, Algorithm Design Manual, and Mind, Body, World - Foundations of Cognitive Science, using embedding model and got top passage among all embeddings on finding cosine similarity for a specific query.
# Model choices
## [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
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
- better scores on same passage from same query using torch.cosine_similarity than from utils.cos_sim. (Why?)
## Latency
### Generating Embeddings 
With overlap = 2
- 44.55950 seconds.
- 16.83830 seconds.
- 47.26642 seconds.
- 32.18161 seconds.
Without overlap
- 38.10320 seconds.
- 14.42824 seconds.
- 38.26974 seconds.
- 26.64616 seconds.
## Example
```
Query: How do operating systems use virtualization to manage memory

Results:
Score: 0.8207
To make sure the OS does so, we need some goals to guide us. We have seen these
goals before (think of the Introduction), and we’ll see them again, but they are
certainly worth repeating. One major goal of a virtual memory (VM) system is
transparency2. The OS should implement virtual memory in a way that is invisible
to the running program. Thus, the program shouldn’t be aware of the fact that
memory is virtualized; rather, the program behaves as if it has its own private
physical memory. Behind the scenes, the OS (and hardware) does all the work to
multiplex memory among many different jobs, and hence implements the illusion.
Another goal of VM is efﬁciency. The OS should strive to make the virtualization
as efﬁcient as possible, both in terms of time (i.e., not mak- ing programs run
much more slowly) and space (i.e., not using too much memory for structures
needed to support virtualization).
Page number: 111
```
# Next Steps
- Better ways to process text
- Better ways to embed structured data, for eg. code embeddings
- Metric comparison with existing methods
- Connection to the project: Finetuning separate from routing
- Better input data and generalized implementation (i.e. Langchain)
- think about encodings for adapters
# Caveats
- Text processing looks different for every text corpus (still a textbook here), so building a one-size-fits-all robust system for document processing atleast might not be so straightforward
- Might use a different embedding model for better results
# Deeper Issues
- no code embeddings procedure - harder to find
- images, header, footer, irrelevant information can be removed
- might need to look for a better embedding model
- might need to look for langchain, other systematic frameworks
# Conclusion about RAG
	However, it's costly and challenging to implement in production, in addition to its low precision (misaligned retrieved chunks) and low recall (failure to retrieve all relevant chunks).
So this is a problem with RAG for long context in general, not just my problem. 
