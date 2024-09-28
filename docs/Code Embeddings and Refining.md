- code queries need to be processed separately because those queries need to be encoded by the same code embedding model used to create code database (this is different from the text embedding model).
- code embeddings to be stored in a separate csv because not all pages will have code snippets. 
- so just make a new pages_and_codes, just like pages_and_texts but it doesn't need info other than page_num, and the sentence just above it?? no not needed for now (unless i decide i need code snippet captioning later)
- the model should tokenize \\n to mean newline, so it works.
# New pipeline
- New pages_and_code (can be done in text_to_dataframe function) -> create_code_embeddings -> embeddings/code/.. path
- same similarity search
## Tests to run
- ~~does it calculate embeddings on pages with no code? (can check this from the stat function)~~
	- doesn't because i separated it to not store code embeddings each page
# Outputs
- example of code embedding generation:
```
100%|████████████████████████████████████████████████████████████████████████████████████| 586/586 [00:01<00:00, 499.91it/s]
100%|███████████████████████████████████████████████████████████████████████████████████| 1017/1017 [00:31<00:00, 32.01it/s]
[INFO] Time taken to generate document embeddings: 31.76913 seconds.
Embeddings saved to embeddings/text/operating_systems_three_easy_pieces_embeddings.csv
100%|█████████████████████████████████████████████████████████████████████████████████████| 559/559 [00:06<00:00, 84.58it/s]
[INFO] Time taken to generate code embeddings: 6.61004 seconds.
Embeddings saved to embeddings/code/operating_systems_three_easy_pieces_embeddings.csv
100%|████████████████████████████████████████████████████████████████████████████████████| 652/652 [00:01<00:00, 556.50it/s]
100%|███████████████████████████████████████████████████████████████████████████████████| 1720/1720 [00:38<00:00, 45.10it/s]
[INFO] Time taken to generate document embeddings: 38.14075 seconds.
Embeddings saved to embeddings/text/algorithm-design-manual_embeddings.csv
100%|█████████████████████████████████████████████████████████████████████████████████████| 652/652 [00:06<00:00, 95.43it/s]
[INFO] Time taken to generate code embeddings: 6.83326 seconds.
Embeddings saved to embeddings/code/algorithm-design-manual_embeddings.csv
```
- example of better code search:
```
Query: "Priority Queue"


Top Results:

Result 1:
Score: 0.5138
Source: algorithm-design-manual_embeddings.csv
Page Number: 122
Code Snippet:
while in a
item_type q[PQ_SIZE+1];
int n;
} priority_queue;


Result 2:
Score: 0.3098
Source: algorithm-design-manual_embeddings.csv
Page Number: 125
Code Snippet:
{
int i;
pq_init(q);
pq_insert(q, s[i]);
}
{
int min = -1;
if (q->n <= 0) printf("Warning: empty priority queue.\n");
min = q->q[1];
q->q[1] = q->q[ q->n ];
q->n = q->n - 1;
bubble_down(q,1);
}
return(min);
}


Result 3:
Score: 0.2801
Source: algorithm-design-manual_embeddings.csv
Page Number: 124
Code Snippet:
{
if (q->n >= PQ_SIZE)
printf("Warning: priority queue overflow insert x=%d\n",x);
q->n = (q->n) + 1;
q->q[ q->n ] = x;
bubble_up(q, q->n);
}
}
{
if (pq_parent(p) == -1) return; /* at root of heap, no parent */
if (q->q[pq_parent(p)] > q->q[p]) {
pq_swap(q,p,pq_parent(p));
bubble_up(q, pq_parent(p));
}
}
{
q->n = 0;
}
```
- example of code search:
```
Query: priority queue

Top Results:

Result 1:
Score: 0.5183
Source: test_embeddings.csv
Page Number: 122
Code Snippet:
while in a item_type q[PQ_SIZE+1]; int n; } priority_queue;


Result 2:
Score: 0.5183
Source: test_embeddings.csv
Page Number: 122
Code Snippet:
while in a item_type q[PQ_SIZE+1]; int n; } priority_queue;


Result 3:
Score: 0.3945
Source: test_embeddings.csv
Page Number: 125
Code Snippet:
{ int i; pq_init(q); for (i=0; i<n; i++) pq_insert(q, s[i]); } { int min = -1;
if (q->n <= 0) printf("Warning: empty priority queue.\n"); min = q->q[1];
q->q[1] = q->q[ q->n ]; q->n = q->n - 1; bubble_down(q,1); } return(min); }
```