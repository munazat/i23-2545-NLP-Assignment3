## Transformer-based Review Understanding with RAG Enhanced Explanation Generation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Critical Restrictions](#critical-restrictions)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [System Pipeline](#system-pipeline)
6. [Part A — Encoder Model](#part-a--encoder-model-for-understanding-25-marks)
7. [Part B — Retrieval Module](#part-b--retrieval-module-15-marks)
8. [Part C — Decoder Model](#part-c--decoder-model-for-explanation-generation-25-marks)
9. [Preprocessing Requirements](#preprocessing-requirements)
10. [Hyperparameter Tuning](#hyperparameter-tuning)
11. [Evaluation Requirements](#evaluation-requirements)
12. [Report Requirements](#report-requirements)
13. [Deliverables & Submission](#deliverables--submission)
14. [Marking Rubric Summary](#marking-rubric-summary)
15. [How to Run](#how-to-run)

---

## Project Overview

This assignment implements a three-stage NLP pipeline that:

1. **Encodes** structured information from Amazon product reviews using an encoder-only Transformer (Part A)
2. **Retrieves** semantically similar reviews using vector similarity search (Part B)
3. **Generates** natural language explanations for predicted sentiments using a decoder-only Transformer conditioned on retrieved context (Part C)

The system forms a complete **Retrieval-Augmented Generation (RAG)** pipeline — all three parts are designed to interoperate as one coherent system.

---

## Critical Restrictions

> **Read this section before writing any code.**

The following are **strictly prohibited**:

| Prohibited | Reason |
|---|---|
| `nn.Transformer` | Must implement from scratch |
| `nn.MultiheadAttention` | Must implement from scratch |
| `nn.TransformerEncoder` | Must implement from scratch |
| Any pretrained model (BERT, GPT, T5, etc.) | Must train from scratch |
| Any library abstracting internal Transformer implementation | Implementation must be transparent |

Everything — attention mechanism, multi-head configuration, encoder/decoder blocks — must be hand-coded in PyTorch using only primitive operations (`nn.Linear`, `nn.LayerNorm`, etc.).

---

## Dataset

**Source:** [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)

### Dataset Construction Requirements

| Requirement | Specification |
|---|---|
| Minimum product categories | 3 |
| Reviews per category | ~10,000 – 15,000 |
| Total dataset size | 30,000 – 45,000 samples |
| Required fields per sample | Review text + Star rating (1–5) |

### Files Used

```
Dataset/
├── sports.json.gz        (~66,676 KB)
├── home.json.gz          (~134,890 KB)
├── electronics.json.gz   (~484,233 KB)
├── beauty.json.gz        (~43,769 KB)
└── cellphones.json.gz    (~44,346 KB)
```

Select **at least three** of these categories. Recommended: `sports`, `beauty`, and `cellphones` (or any three that together yield 30k–45k samples after filtering).

### Data Splits

| Split | Percentage |
|---|---|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

> **Important:** Vocabulary must be built using **training data only**. Validation and test sets must use the same vocabulary without contributing new tokens.

---

## Project Structure

```
NLP Assignment 2/
├── i23XXXX-NLP-Assignment2.ipynb     ← Main Jupyter Notebook (all code)
├── README.md                          ← This file
├── Dataset/
│   ├── sports.json.gz
│   ├── home.json.gz
│   ├── electronics.json.gz
│   ├── beauty.json.gz
│   └── cellphones.json.gz
├── models/
│   ├── encoder_model.pt               ← Trained encoder weights
│   └── decoder_model.pt               ← Trained decoder weights
└── results/
    ├── train_embeddings.npy           ← Encoder embeddings for training set
    ├── train_labels.npy               ← Corresponding labels
    ├── encoder_metrics.json           ← Evaluation results for Part A
    ├── decoder_metrics.json           ← Evaluation results for Part C
    └── learning_curves/               ← Training/validation loss plots
```

---

## System Pipeline

```
[Raw Review Text]
       │
       ▼
[Preprocessing Pipeline]
  - Text cleaning
  - Tokenization
  - Vocab construction (train only)
  - Numericalization
  - Padding / Truncation
       │
       ▼
┌──────────────────────────────┐
│  PART A: Encoder-Only        │
│  Transformer (from scratch)  │
│                              │
│  Outputs:                    │
│  ├─ Sentiment label (3-way)  │
│  ├─ Derived feature          │
│  └─ Review embedding vector  │
└──────────────────────────────┘
       │                  │
       │ embeddings        │ labels
       ▼                  ▼
┌──────────────┐    ┌──────────────────────────────┐
│  PART B:     │    │  PART C: Decoder-Only         │
│  Retrieval   │───▶│  Transformer (from scratch)   │
│  Module      │    │                               │
│              │    │  Input: review + sentiment +  │
│  Cosine sim  │    │  derived feature + top-k      │
│  Top-k       │    │  retrieved reviews            │
│  similar     │    │                               │
│  reviews     │    │  Output: 1-2 sentence natural │
└──────────────┘    │  language explanation          │
                    └──────────────────────────────┘
```

---

## Part A — Encoder Model for Understanding (25 Marks)

### Task Definition

An **encoder-only Transformer** that jointly performs:

1. **Sentiment Classification (3-class)**
   - Ratings 1–2 → `Negative`
   - Rating 3 → `Neutral`
   - Ratings 4–5 → `Positive`

2. **Derived Feature Prediction** *(student-defined)*
   - Must be a meaningful property predictable from text alone
   - Must be clearly motivated and documented in report
   - Examples: review length category (short/medium/long), helpfulness score estimate, product category, presence of comparison language

3. **Review Embedding**
   - A fixed-dimensional dense vector representation of each review
   - Must be saved to disk after training (used in Part B)

### Architecture Requirements

| Component | Requirement |
|---|---|
| Input representation | Numerical token sequences; vocab from training only |
| Attention mechanism | Manually implemented; queries, keys, values produce meaningful alignments |
| Multi-head attention | Multiple heads operating independently; outputs correctly concatenated |
| Encoder block | Layer normalization + feed-forward sublayer + residual connections |
| Multi-task heads | Shared encoder trunk, separate output heads for each task |
| Combined loss | Weighted sum of task losses; weights are design choices to document |

### Outputs

- Sentiment label per review
- Derived feature prediction per review
- Fixed-dimensional embedding per review (saved for retrieval)

### Part A Marking Rubric

| Component | Marks | Key Criteria |
|---|---|---|
| Input representation | 3 | Correct numericalization; vocab from training only |
| Attention mechanism | 4 | Correct Q/K/V; meaningful alignments |
| Multi-head configuration | 4 | Independent heads; correct output combination |
| Encoder block design | 4 | Norm + FFN + residual all present and correct |
| Multi-task classification | 4 | Both tasks predicted from shared encoder |
| Training pipeline | 3 | Optimizer, schedule, loss curves present |
| Evaluation and embeddings | 3 | Metrics for both tasks; embeddings saved |
| **Total** | **25** | |

---

## Part B — Retrieval Module (15 Marks)

### Task Definition

Given a test review's encoder embedding, find the **top-k most similar training reviews** to serve as context for the decoder.

### Implementation Requirements

1. **Embedding Storage**
   - Store encoder embeddings for the entire training corpus
   - Must be efficiently accessible at inference time (e.g., NumPy array, FAISS index)

2. **Query Construction**
   - For each test review, build a query vector from its encoder output

3. **Similarity Search**
   - Compute similarity between the query and all stored training embeddings
   - Default metric: cosine similarity
   - Retrieve and rank top-k results; k must be configurable

4. **Integration**
   - Retrieved review texts (and optionally their labels) are passed as context to the decoder

### Report Analysis Required

- Examples of query reviews alongside retrieved matches
- Discussion of whether matches are semantically meaningful
- Reflection on limitations and downstream impact
- Comment on similarity metric choice and its impact
- Exploration of how varying k affects relevance and diversity
- Potential improvements (better embeddings, FAISS, approximate nearest neighbor)
- Relation between retrieval quality and overall RAG performance

### Part B Marking Rubric

| Component | Marks | Key Criteria |
|---|---|---|
| Embedding storage and indexing | 4 | All training embeddings stored; efficient access |
| Query construction | 3 | Meaningful query built from encoder output |
| Similarity search | 4 | Correct ranking and retrieval |
| Retrieval quality and integration | 4 | Relevant results; context passed to decoder correctly |
| **Total** | **15** | |

---

## Part C — Decoder Model for Explanation Generation (25 Marks)

### Task Definition

A **decoder-only Transformer** that generates a 1–2 sentence natural language explanation for why a review carries its predicted sentiment.

### Input Construction

The decoder takes a structured concatenation of four elements:

1. Original review text
2. Predicted sentiment label (from Part A)
3. Predicted derived feature value (from Part A)
4. Top-k similar reviews retrieved in Part B

A clear, consistent template must be defined for combining these into a single input sequence. The template must be documented in the report.

**Example template:**
```
[REVIEW]: <review text> [SENTIMENT]: <label> [FEATURE]: <value>
[CONTEXT]: <retrieved_review_1> ... <retrieved_review_k>
[EXPLANATION]:
```

### Architecture Requirements

| Component | Requirement |
|---|---|
| Causal masking | Model cannot attend to future positions during training or inference |
| Autoregressive generation | Text generated token-by-token; previous tokens condition each step |
| Decoder blocks | Stacked with correct sublayers and residual connections |
| Training objective | Language modeling (next-token prediction / cross-entropy) |
| Termination | Generation must terminate correctly (e.g., `<EOS>` token or max length) |

### Evaluation Requirements

- **Perplexity** on the held-out test set
- **At least 5 generated examples** with written commentary on quality, coherence, and relevance
- **RAG ablation study**: compare full system (with retrieval) vs. baseline (no retrieval context); document the improvement

### Part C Marking Rubric

| Component | Marks | Key Criteria |
|---|---|---|
| Input construction | 3 | All 4 inputs combined coherently |
| Decoder attention | 5 | Causal masking correctly prevents future-token attention |
| Autoregressive generation | 5 | Token-by-token generation; correct termination |
| Decoder block design | 4 | Stacked blocks with correct sublayers + residuals |
| Quantitative evaluation | 4 | Perplexity correctly computed and reported |
| Qualitative evaluation | 2 | ≥5 examples analyzed for coherence and relevance |
| RAG ablation study | 2 | Baseline vs. full system comparison documented |
| **Total** | **25** | |

---

## Preprocessing Requirements

All preprocessing must be implemented manually (no Hugging Face tokenizers, spaCy, etc. unless explicitly approved). The pipeline must include:

| Step | Requirement |
|---|---|
| Text cleaning | Remove HTML tags, special characters, normalize whitespace; apply where necessary |
| Tokenization | Word-level or subword; must be implemented yourself |
| Vocabulary construction | Built from **training data only**; include `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>` special tokens |
| Numericalization | Convert tokens to integer indices |
| Padding | Pad sequences to a fixed maximum length |
| Truncation | Truncate sequences exceeding maximum length |

Every step must be described clearly in the report.

---

## Hyperparameter Tuning

You must experiment with and document the following:

| Hyperparameter | What to Vary |
|---|---|
| Learning rate | Try at least 3 values |
| Number of Transformer layers | e.g., 2, 4, 6 |
| Hidden dimension size | e.g., 128, 256, 512 |
| Number of attention heads | Must divide hidden dim evenly |
| Dropout rate | e.g., 0.1, 0.2, 0.3 |
| Batch size | e.g., 32, 64, 128 |
| Sequence (max token) length | e.g., 64, 128, 256 |
| k (number of retrieved examples) | e.g., 1, 3, 5 |

Every configuration tried must be logged with its effect on validation performance. The report must reflect genuine exploration, not just a final result.

---

## Evaluation Requirements

### Part A
- Accuracy, Precision, Recall, F1-score for sentiment classification
- Appropriate metric for the derived feature (accuracy if classification, RMSE/MAE if regression)
- Learning curves (train and validation loss per epoch)

### Part B
- Qualitative examples showing query → retrieved results
- Discussion of retrieval relevance

### Part C
- Perplexity on test set
- ≥5 generated explanation examples with written analysis
- Ablation table comparing with-retrieval vs. without-retrieval

---

## Report Requirements

**Length:** 3–5 pages

The report must cover:

- [ ] Overall system design and methodology
- [ ] Justification of all design decisions for each part
- [ ] Preprocessing pipeline description (every step)
- [ ] Justification of the derived feature choice (Part A)
- [ ] Evaluation results with tables and plots for all metrics
- [ ] Hyperparameter tuning log with analysis of results
- [ ] RAG ablation study comparing with and without retrieval
- [ ] Reflection on retrieval quality and its impact
- [ ] Input template definition for the decoder (Part C)

---

## Deliverables & Submission

### Files to Submit

| Deliverable | Location | Notes |
|---|---|---|
| Jupyter Notebook | Root of repo | Named `i23XXXX-NLP-Assignment2.ipynb` |
| Encoder weights | `models/encoder_model.pt` | |
| Decoder weights | `models/decoder_model.pt` | |
| Training embeddings | `results/train_embeddings.npy` | From Part A encoder |
| Evaluation outputs | `results/` | Metrics, plots |
| Report (PDF) | Root of repo | 3–5 pages |

### Notebook Naming Convention
```
i23XXXX-NLP-Assignment2.ipynb
```
Replace `XXXX` with your student ID.

### GitHub Submission Rules

- All submissions via **GitHub with incremental commits**
- Commit history will be reviewed as part of evaluation
- Commits must be logical, progressive, and have meaningful messages
- Do not push a single massive commit at the end

---

## Marking Rubric Summary

| Component | Marks |
|---|---|
| Part A: Encoder Model | 25 |
| Part B: Retrieval Module | 15 |
| Part C: Decoder & RAG Pipeline | 25 |
| Report Quality | 10 |
| GitHub & Incremental Commits | 5 |
| **TOTAL** | **80** |
| Bonus: UI/Visualization OR Advanced Features | Up to +5 |

> Note: The bonus is awarded on top of earned marks. The final recorded mark will not exceed 80.

---

## How to Run

> Place this at the top of your notebook as clear execution instructions.

```
Prerequisites:
  - Python 3.9+
  - PyTorch >= 2.0
  - NumPy, Pandas, Matplotlib, scikit-learn, tqdm

Dataset Setup:
  - Place all .json.gz files inside the Dataset/ folder
  - Run all cells top-to-bottom in the notebook

Directory Setup:
  - models/ and results/ directories are created automatically by the notebook
```
