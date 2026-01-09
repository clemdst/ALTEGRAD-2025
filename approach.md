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
Here is a corrected and professional revision of your text, incorporating the technical details and results we discussed today.

## Phase 3: The Matryoshka Representation Learning (MRL) Approach

**Approach:**
To improve retrieval efficiency and robustness, we implemented a Matryoshka Representation Learning strategy. Instead of training for a single fixed output dimension, this loss function forces the model to learn a hierarchical representation where the first  dimensions (e.g., 64, 128, 256) are independently capable of accurate retrieval. This allows us to "slice" the embedding vector at inference time to balance speed and accuracy.

**Results:**
The training yielded the following Best Scores:

* **MRR@64:** 0.5235
* **MRR@128:** 0.5748
* **MRR@256:** 0.5996 (Peak)
* **MRR@512:** 0.5867
* **MRR@768:** 0.5970

**Analysis:**
The results clearly demonstrate that the "sweet spot" for our data lies at **256 dimensions**, where the model actually outperformed the full 768-dimensional vector (0.5996 vs 0.5970). This suggests that the intrinsic dimensionality of the chemical space is lower than the default SciBERT output. However, since our baseline models often operated near this dimensionality already, this phase primarily optimized efficiency rather than yielding a massive jump in raw performance.

## Phase 4: Two-Stage Retrieval (Retrieve & Rerank)

**Approach:**
We constructed a pipeline consisting of two distinct stages:

1. **Retriever:** We utilized the Matryoshka model (sliced to the optimal 256 dimensions) to rapidly retrieve the top-20 candidates.
2. **Reranker:** We trained a specialized Residual Reranker (a lighter GNN with a cross-attention mechanism) to re-order these top-20 candidates based on a full-dimensional comparison.

**Results:**
Despite the theoretical advantage of reranking, we observed only a marginal improvement in our leaderboard score, moving from **0.56304** to **0.56776**.

**Analysis:**
This plateau indicated that our bottleneck was not the ranking logic, but the fundamental quality of the embeddings themselves. The static text embeddings were "upper-bounding" our performance; no amount of reranking could fix the fact that the graph and text vector spaces were not perfectly aligned. We concluded that we needed to fundamentally revise how we embed both modalities.

## Phase 5: The Dual-Tower Graph Transformer (SOTA)

**Approach:**
To break the performance ceiling, we moved to a **Dual-Tower (Dual-Encoder)** architecture that trains both modalities jointly:

1. **Graph Tower (The Upgrade):** We replaced the standard GINEConv with a **Graph Transformer**. Unlike GINE, which is limited to local neighborhoods, the Transformer uses self-attention with edge features, allowing it to capture global molecular structures and long-range dependencies between atoms.
2. **Text Tower (The Adapter):** Instead of treating SciBERT embeddings as static "ground truth," we added a trainable MLP projection layer. This acts as an adapter, learning to map the generic scientific text into our specific chemical vector space.

**Inference Mechanism:**
At inference time, we perform a "Library Upgrade": we pass all training and validation descriptions through the trained Text Tower once to project them into the shared space. We then pass the test graphs through the Graph Tower and perform retrieval against this upgraded library.

**Results:**
This approach successfully aligned the vector spaces, allowing the model to highlight relevant chemical terms in the descriptions while ignoring generic text. This yielded our significant breakthrough, achieving a score of **0.60162**.

## Phase6: Domain-Specific Chemical Embeddings (ChEmbed)

**1. The Problem:**
While SciBERT was a significant upgrade over general English models, it remains a broad-spectrum scientific encoder. It understands "science" generally but lacks the high-resolution, specialized understanding of chemical nomenclature and the dense relationship between molecular descriptors and functional properties required for precise captioning. To achieve high semantic accuracy (BERTScore), the model needs a deeper grasp of chemical-specific tokens.


**2. The Solution:**

We replaced the Text Tower’s backbone with BASF-AI/ChEmbed-base:


**Chemistry-Native:** Unlike SciBERT, this model is pre-trained specifically on chemical structures and specialized text, providing an embedding space that naturally aligns with the molecular graph domain.

**Reduced Adapter Strain:** By starting with a more relevant text representation, our projection layers (the "adapters") required less transformation to map generic scientific text into our specific chemical vector space.


**Enhanced Semantic Mapping:** This model better handles the "sequential, semantic structure" of natural language as it relates to chemistry, bridging the gap more effectively than general-purpose models.


## Future suggestions: 

Implement Hard Negative Mining: Stop using random negatives. We should train the model on the "hardest" wrong answers—molecules that are structurally similar but have different descriptions—to force it to learn fine-grained chemical distinctions.

Upgrade to Domain-Specific Encoders: Replace the generic SciBERT with ChEmbed or MolT5. Starting with models pre-trained specifically on chemical reactions and SMILES syntax will give us a much stronger baseline than general scientific text.

Adopt "Late Interaction" (ColBERT-style): Move beyond compressing everything into a single vector. By matching individual atoms in the graph directly to relevant words in the description (e.g., matching a "chloro" group to the word "chlorine"), we can bypass the information bottleneck of our current dual-tower approach.


