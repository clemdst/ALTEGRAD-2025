# ALTEGRAD 2025: Molecular Graph Captioning

This repository contains the implementation for the ALTEGRAD 2025 Data Challenge. The goal is to bridge the gap between molecular graphs and natural language by developing models that generate or retrieve precise text descriptions for chemical structures.

## 📂 Project Structure

* `data/`: Raw and preprocessed graph data (`train`, `val`, `test`).
* `embeddings/`: Destination folder for generated node/graph embeddings.
* `embeddings_scripts/`: Scripts to generate embeddings (e.g., GCN, BERT, ChemBERTa).
* `strategy/`: Core logic for training generative models and running retrieval algorithms.
* `models/`: Model architecture definitions.
* `utils/`: Helper functions for data loading and evaluation.
* `results/`: Output folder for predictions and submission files.

---


## 🚀 Installation
<details>
<summary><strong> (Click to expand)</strong></summary>
### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/ALTEGRAD-2025.git](https://github.com/your-username/ALTEGRAD-2025.git)
cd ALTEGRAD-2025
```

### 2. Install Requirements
Ensure you are using a Python environment (Python 3.8+ recommended). Install the necessary dependencies:

```Bash

pip install -r requirements.txt

```

## ⚙️ Step 1: Generating Embeddings
Before running any training or retrieval strategies, you must pre-compute the embeddings for the molecular graphs and text descriptions.

Navigate to the embedding generation scripts:


⚠️ Important: Hugging Face Clearance for chembed
This project utilizes ChemBED (or similar gated models like ChemBERTa/MolT5) which requires authentication.

Hugging Face Account: You must have a Hugging Face account.

Model Clearance: Specific models are gated. Visit the model card on Hugging Face and accept the license terms to gain access.

Authentication: Log in via your terminal using your access token:

```Bash
cd embeddings_scripts
huggingface-cli login
```
### Paste your User Access Token when prompted
Running the Scripts
Run the generation scripts to populate the embeddings/ directory.

``` Bash
# Example: Generate graph embeddings
python generate_graph_embeddings.py

# Example: Generate text/chemical embeddings (requires HF login)
python generate_chembed.py
```

## 🧠 Step 2: Running Strategies
Once the embeddings are ready, you can execute the main strategies located in the strategy/ folder. This includes training generative models or running the baseline retrieval algorithms.

Train a Model
To train a new model (e.g., a Graph-to-Text generator):

```Bash
# Example command
python strategy/train_generative.py
```

Run Retrieval Baseline
To execute the embedding-similarity retrieval algorithm (matching test graphs to training captions):

```Bash
# Example command
python strategy/run_retrieval.py
```

## 📊 Evaluation

Results and predictions will be saved to the results/ directory. Formatting of the output files adheres to the Kaggle submission standards.
</details>

## 📄 Project Report

You can view our detailed final report, including the analysis of the Generative Frontier and Cross-Modal Attention experiments, by clicking the link below:

[**View Full Project Report (PDF)**](./ALTEGRAD_report.pdf)
