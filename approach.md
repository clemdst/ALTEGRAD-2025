# Project Update: Molecular Retrieval Model Improvements

We improved our molecular retrieval model in two major phases. This document outlines the changes made to the architecture and data processing to increase retrieval accuracy.

## Phase 1: Upgrading the "Vision" (GCN $\rightarrow$ GINE)

Our first goal was to make the model actually understand chemistry, rather than just the geometric shape of the graph.

### 1. The Problem (Old Model)
* **Blind to Atoms:** The original GCN treated every node exactly the same. To the model, a Carbon atom and an Oxygen atom looked identical.
* **Blind to Bonds:** It completely ignored the edges. It did not distinguish between single, double, or aromatic bonds.
* **Result:** The model only saw the "skeleton" of the molecule but missed the chemical properties.

### 2. The Solution (New Model)


We switched to a **GINE (Graph Isomorphism Network with Edge features)** architecture:
* **Atom Features:** We now provide specific inputs for every atom type. The model knows specifically what element each node is.
* **Edge Features:** We now include bond attributes. The model understands how atoms are connected (single vs. double bonds).
* **Better Training:** We switched from **MSE Loss** (forcing numbers to match) to **Contrastive Loss**. This teaches the model to look at a batch of descriptions and pick the *one* correct match for the graph.

**Phase 1 Best Result (MRR):** 

- *Validation Score:*  {'MRR': 0.3549558222293854}

- *Public Score:* 0.52324

---

## Phase 2: Upgrading the "Language" (SciBERT)

Once the model could "see" the chemistry, we needed to ensure it understood the text descriptions correctly.

### 1. The Problem
We were using standard text embeddings trained on general English (Wikipedia, Books). These models often struggle with specific technical terms like "aromatic ring," "derivative," or "inhibitor."

### 2. The Solution


We switched to **SciBERT**:
* **Science-Native:** This model was trained on millions of scientific papers.
* **Result:** It generates much smarter embeddings for our descriptions, giving the GNN a higher-quality target to learn from.

**Phase 2 Best Result (MRR):** 

- *Validation Score:*  {'MRR': 0.5981288552284241,}

- *Public Score:* 0.56304


## Future Work: Two-Stage Reranking (Suggestions)

We are currently exploring a **"Reranking"** strategy to further improve accuracy.

### The Idea
Instead of relying on just one model, we want to split the process into two stages:
1.  **The Retriever (Current Model):** Quickly selects the "Top-10" likely matches.
2.  **The Reranker (New Model):** Takes those 10 candidates and looks at them very closely (combining Graph and Text features together) to find the absolute best match.



### Current Status
We have implemented a prototype where a second model re-orders the Top-10 choices.
* **Initial Results:** The performance is currently lower than the single model.
* **Next Steps:** We need to investigate training stability (preventing the new model from forgetting what the old model learned) and improve how we handle "negative" samples during training.
