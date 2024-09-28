# Strategies for score improvement
- Trying better code embedding models
	- uniXcoder trained on LoRA adapters (like jina_embedding_v3)
- Maybe code embedding models should be multimodal, such that alongwith code snippets, user can also input descriptive captions (or keywords) about code. Some keywords encoded with the code snippet will improve semantic search. Will research to see if this has been done before.
- Min token limit on code?
- Trying on more code-heavy textbooks
# Some good outputs: 
(scores may be low but code snippet relevance to query is pretty high)
```
Query: "Priority Queue"

Top Results:

Result 1:
Score: 0.5231
Source: algorithm-design-manual_embeddings.csv
Page Number: 122
Code Snippet:
while in a
item_type q[PQ_SIZE+1];
int n;
}
priority_queue;

Result 2:
Score: 0.3279
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
Score: 0.2893
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
if (pq_parent(p) == -1) return;
/* at root of heap, no parent */
if (q->q[pq_parent(p)] > q->q[p]) {
    pq_swap(q,p,pq_parent(p));
    bubble_up(q, pq_parent(p));
}
}
{
q->n = 0;
}
```

# Interesting Results
Since code snippets are chunked page-wise, the top 2 results are likely the correct code snippet separated across the 2 pages. Additionally, not sure how, but result # 3 is still valid, since no edges also means no cycles in a graph.

Solution: Maybe a better chunking strategy for code snippets rather than page-wise separation.
```
Query: "Detect cycles using DFS"

Top Results:

Result 1:
Score: 0.4197
Source: algorithm-design-manual_embeddings.csv
Page Number: 193
Code Snippet:
{
    int i;
    init_stack(&sorted);
    if (discovered[i] == FALSE)
dfs(g,i);
    print_stack(&sorted);
}
if there is a directed
= (
= (

Result 2:
Score: 0.4146
Source: algorithm-design-manual_embeddings.csv
Page Number: 192
Code Snippet:
{
}
{
}
{
    push(&sorted,v);
}
{
    int class;
    class = edge_classification(x,y);
    if (class == BACK)
printf("Warning: directed cycle found, not a DAG\n");
}

Result 3:
Score: 0.4143
Source: algorithm-design-manual_embeddings.csv
Page Number: 337
Code Snippet:
= (
if there are no edges (
```

# Victims of incorrect code snippet detection
The current code uses keyword search on exact first word of a particular span to check if this is the entire text block is a code. Sentences ending with ';' are checked, those beginning with indents are checked, invalid mini phrases are removed and texts with fonts 'Courier', 'Mono' are also checked for code snippet detection.
```
Query: "Prim's Algorithm"

Top Results:

Result 1:
Score: 0.3540
Source: algorithm-design-manual_embeddings.csv
Page Number: 45
Code Snippet:
Problem Size
Best Case
Average Case
Worst  Case
of Steps
Number

Result 2:
Score: 0.3120
Source: algorithm-design-manual_embeddings.csv
Page Number: 436
Code Snippet:
class provides arbitrary-precision analogues to all

Result 3:
Score: 0.2985
Source: algorithm-design-manual_embeddings.csv
Page Number: 605
Code Snippet:
Even the most elementary-sounding bin-packing problems are NP-complete;
```

