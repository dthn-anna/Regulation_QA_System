# Regulations QA System

**Vietnamese Question Answering for Higher-Education Regulations with Hybrid Retrieval-Augmented Generation**

This repository contains the code for the paper *"Vietnamese Question Answering System for Higher Education Regulations Using Hybrid Retrieval-Augmented Generation"* (RIVF 2026). The authors are from the University of Information Technology and the University of Economics and Law, VNU-HCM.

Students ask questions in natural Vietnamese about academic regulations, tuition, commendations, disciplinary rules and similar topics. The system retrieves the relevant regulatory provision and generates an answer based only on that provision.

---

## Architecture


```
Question ─► ┌──────────────── Hybrid Retriever ────────────────┐
            │  BM25 (pyvi tokenizer)  +  Dense bi-encoder       │
            │        └──── Multiplication-Ensemble ────┘        │
            └───────────────────────┬──────────────────────────┘
                                    │ top-k provision
                                    ▼
                    Generator (fine-tuned Qwen3) ─► Answer + citations
```

- **Sparse:** BM25 with `k1 = 0.9` and `b = 0.9`, chosen by grid search on the dev set.
- **Dense:** bi-encoders fine-tuned by **knowledge distillation** from the cross-encoder `BGE-Reranker-v2-M3`, using a Margin-MSE loss. Hard negatives are mined in four steps:
  1. Take the BM25 top-50 for each query.
  2. Rerank these passages with the cross-encoder.
  3. Drop the highest-ranked passages, which are likely unlabeled positives.
  4. Keep the next 8 passages as hard negatives.
- **Hybrid:** we compare score-based Multiplication-Ensemble with rank-based Reciprocal Rank Fusion. The weight `α` is chosen by grid search over `[0, 1]` in steps of 0.05.
- **Generator:** we use lightweight open models: Qwen3-0.6B, Qwen3-1.7B and Gemma3-270M. Each model is fine-tuned on ViRHE4QA. It receives the **top-1** retrieved passage as context and is instructed to answer only from that evidence.

## Main results (ViRHE4QA test set)

**Retrieval:** Multilingual-E5-large is the dense component.

| Method | P@1 | MRR@5 | nDCG@5 | P@10 |
|---|---|---|---|---|
| BM25 | 0.7128 | 0.7916 | 0.8212 | 0.9464 |
| Bi-Encoder (m-E5-large, fine-tuned) | 0.7338 | 0.8117 | 0.8393 | 0.9541 |
| RRF | 0.7728 | 0.8555 | 0.8835 | **0.9818** |
| **Multiplication-Ensemble** | **0.8152** | **0.8741** | **0.8968** | 0.9791 |
| *R2GQA hybrid (reference)* | *0.7274* | – | – | *0.9652* |

**End-to-end RAG:**

| Generator | F1 | BLEU-1 | BLEU-4 | ROUGE-L | BERTScore |
|---|---|---|---|---|---|
| **Qwen3-1.7B** | **75.32** | **70.99** | **63.21** | **70.29** | **89.69** |
| Qwen3-0.6B | 74.76 | 67.76 | 59.95 | 69.76 | 89.50 |
| Gemma3-270M | 66.78 | 58.41 | 50.16 | 61.96 | 86.99 |
| *R2GQA* | – | *58.60* | *50.20* | *63.34* | *77.13* |

The paper has the full tables: results for every dense model and every cutoff K, plus generator scores with gold context.

---

## Repository structure

```
Regulations_QA_System/
├── 02_Retriever/
│   ├── 01_Processing/Preprocessing.ipynb        # normalization, stable IDs, deduplication
│   ├── 02_Baseline/
│   │   ├── Sparse_Retriever/BM25.ipynb          # BM25 + k1/b grid search
│   │   └── Dense_Retriever/03a–03e_*.ipynb      # zero-shot baselines: bge-m3, m-E5-large,
│   │                                            #   vietnamese-bi-encoder, Qwen3-Embedding-0.6B/4B
│   ├── 03_Finetuning/
│   │   ├── 4a_Evaluate_teacher_model.ipynb      # evaluate the cross-encoder teacher
│   │   ├── 4b_Teacher_scoring.ipynb             # score (query, passage) pairs with the teacher
│   │   ├── Mine hard negative.ipynb             # hard-negative mining
│   │   └── 4c–4g_Finetune_*.ipynb               # distill into each bi-encoder
│   ├── 04_Hybrid/04_hybrid_retrieval.ipynb      # fuse BM25 + dense, tune α, export serving config
│   ├── Precision@.ipynb                         # recompute P@k / MRR / nDCG, compare with R2GQA
│   └── retriever_common.py                      # shared helpers (corpus encoding, chunk max-pooling…)
│
├── 03_Generator/
│   ├── 03_Generator_Qwen3-0.6B.ipynb            # fine-tuning + generation
│   ├── 03_Generator_Qwen3-1.7B.ipynb
│   ├── 03_Gemma3.ipynb
│   ├── 03_Generator.ipynb                       # original / generic notebook
│   ├── End-to-end Evaluation.ipynb              # full-pipeline evaluation and error analysis
│   └── generator_common.py
│
└── Demo/                                        # local Streamlit chatbot
    ├── app_streamlit.py                         # UI only (no model logic or parameters)
    ├── rag_pipeline.py                          # CorpusStore, BM25/Dense/HybridRetriever, Generator, RAGPipeline
    ├── retriever_common.py
    ├── settings.py                              # all paths and serving parameters
    ├── verify_local.py                          # checks that local results match the experiments
    ├── config.toml                              # Streamlit theme
    ├── requirements.txt
    └── README_LOCAL.md                          # detailed guide for moving the system from Drive to local (Vietnamese)
```

## Data

The experiments use **ViRHE4QA** (Do et al., *R2GQA*, Artificial Intelligence and Law, 2025). ViRHE4QA is a public QA dataset on Vietnamese higher-education regulations. The corpus has about 294 regulatory passages. Datasets and model checkpoints are **not** included in this repository.

---

## Reproducing the experiments

The notebooks are written for **Google Colab**. They read and write under `/content/drive/MyDrive/Legal_Sytem/`. If you use a different location, edit the path variables in the first cell of each notebook.

Run the notebooks in this order:

1. `02_Retriever/01_Processing/Preprocessing.ipynb` builds `corpus.parquet`, `qa_master.parquet` and the queries/qrels files.
2. `02_Baseline/Sparse_Retriever/BM25.ipynb` and the `Dense_Retriever/03*` notebooks produce the baselines.
3. In `03_Finetuning/`, run `4a` → `4b` → `Mine hard negative` → `4c`–`4g`. The last step fine-tunes each bi-encoder.
4. `04_Hybrid/04_hybrid_retrieval.ipynb` fuses the rankings, selects the best configuration and exports `retriever_serving.json`.
5. `03_Generator/03_Generator_*.ipynb` fine-tunes the generators. Each checkpoint is saved together with its `prompt_config.json`.
6. `03_Generator/End-to-end Evaluation.ipynb` evaluates the full system.

Hardware: fine-tuning Qwen3-Embedding-4B and Qwen3-1.7B needs a GPU. We recommend at least 24 GB of VRAM, or an A100 on Colab.

---

## Running the demo locally

### 1. Install

```bash
cd Demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Important:** pin the exact `pyvi` version used in the experiments. A different `pyvi` version tokenizes text differently. This changes the BM25 index and the rankings without raising any error.

### 2. Get the models and data

Copy the following files from Drive. [`Demo/README_LOCAL.md`](Demo/README_LOCAL.md) lists them in detail and includes a command for packaging them.

```
Demo/
├── config/retriever_serving.json            # from 02_Retriever/serving/
├── data/
│   ├── corpus.parquet                       # required
│   ├── qa_master.parquet                    # for verify_local.py and threshold calibration
│   └── references/                          # reference rankings/predictions from Colab
├── models/
│   ├── retriever/multilingual-e5-large/     # fine-tuned bi-encoder checkpoint
│   └── generator/Qwen3-0.6B/                # generator checkpoint + prompt_config.json
└── cache/                                   # auto-generated corpus vectors
```

The best generator in the paper is Qwen3-1.7B. To use it:

1. Put its checkpoint in `models/generator/Qwen3-1.7B/`.
2. Set `GENERATOR_RUN_NAME` in `settings.py` to `Qwen3-1.7B`.

All model changes go in `settings.py`. You don't need to edit `app_streamlit.py`.

In `retriever_serving.json`, the `model_save_dir` key still points to Drive. At runtime, `settings.DENSE_MODEL_DIR` overrides it.

### 3. Verify and run

```bash
python settings.py                   # lists missing files (OK / THIẾU = missing)
python verify_local.py --quick       # compare local retrieval with the experiment rankings
python verify_local.py               # check both retriever and generator
python verify_local.py --calibrate   # compute the out-of-scope abstention threshold
streamlit run app_streamlit.py
```

After running `--calibrate`, copy the printed value into `ABSTAIN_THRESHOLD` in `settings.py`. The system will then decline questions that are unrelated to the regulations.

### Serving parameters (`settings.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `TOP_K` | `1` | Number of provisions passed to the prompt. The generator was fine-tuned with exactly one passage, so a higher value moves it outside its training distribution. |
| `DISPLAY_K` | `5` | Number of provisions shown as citations. This does not affect the prompt. |
| `ABSTAIN_THRESHOLD` | `None` | Retriever-score threshold below which the system declines to answer. |
| `DEVICE` | auto | `cuda` if a GPU is available, otherwise `cpu`. |

To change the project root, set the `Regulation_QA_System` environment variable.

### Notes for CPU-only machines

- Models run in `float32` instead of `bfloat16`. Because of this, greedy decoding may differ slightly on a few samples.
- Qwen3-0.6B generates about 5–15 tokens per second. One answer takes roughly 10–30 seconds.
- The first run encodes the whole corpus, which takes a few minutes. The vectors are cached and keyed by the checkpoint fingerprint and the list of `doc_id`s, so later runs start quickly.
- If the retriever checkpoint is missing, `SentenceTransformer` downloads the base model from HuggingFace without any warning. `verify_local.py` catches this because the rankings will not match.

---

## Citation

```bibtex
@inproceedings{duong2026regqa,
  title     = {Vietnamese Question Answering System for Higher Education Regulations
               Using Hybrid Retrieval-Augmented Generation},
  author    = {Duong, Nhung Thi-Hong and Luu, Son T. and Nguyen, Nghia Hieu and
               Tran, Khiem Vinh and To, Huy Quoc and Huynh, Tin Van and Le, Nam Vu and
               Nguyen, Ngan Luu-Thuy and Van Nguyen, Kiet},
  booktitle = {Proceedings of the RIVF International Conference on Computing and
               Communication Technologies (RIVF)},
  year      = {2026}
}
```

If you use the ViRHE4QA dataset, please also cite:

```bibtex
@article{do2025r2gqa,
  title   = {R2GQA: Retriever-Reader-Generator Question Answering System to Support
             Students Understanding Legal Regulations in Higher Education},
  author  = {Do, T. P. P. and Cao, N. D. D. and Tran, K. Q. and Van Nguyen, Kiet},
  journal = {Artificial Intelligence and Law},
  pages   = {1--46},
  year    = {2025}
}
```

## Acknowledgments

This research is funded by Vietnam National University Ho Chi Minh City (VNU-HCM) under grant number DS2025-26-01.

## Contact

Kiet Van Nguyen (corresponding author): kietnv@uit.edu.vn
