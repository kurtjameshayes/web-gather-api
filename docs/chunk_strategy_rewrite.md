## Rewrite the chunking, indexing and compliance strategy.

# Chunking/Indexing for Process for statutes will be:
1) Save statute full text to statutes collection.
2) Parse into logical sections. Save to statute_sections collection.
3) For each section, parse into subsections. Save to statute_subsections collection.
4) Create embeddings for each subsection within statute_subsection collection.
5) Create vector index on the statute_subsection collection.


# Chunking/Indexing for Process for policies will be:
1) Save policy full text to policies collection.
2) Chunk by page size with 20% overlap.
3) Create embeddings and vector index. Save to policy_embeddings.


# Gap Analysis
For each statute subsection, 


