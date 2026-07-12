<div align="center">

# 📘⚜️ The AI/ML Interview Book

### From Scratch to Pro — Theory, Code, and Interview Q&A

[![Chapters](https://img.shields.io/badge/Chapters-26%20Drafted-blue?style=for-the-badge)]()
[![Q&A](https://img.shields.io/badge/Q%26A%20Boxes-730+-gold?style=for-the-badge)]()
[![Code](https://img.shields.io/badge/Code%20Listings-204-blueviolet?style=for-the-badge)]()
[![Figures](https://img.shields.io/badge/Figures-57-teal?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)]()

**A comprehensive, production-grade technical book spanning 39 chapters across 8 parts — covering Mathematics, Classical ML, Deep Learning, NLP, LLMs, System Design, and Interview Prep. Every algorithm is implemented from scratch in pure Python/NumPy, every claim is backed by executed code, and every chapter includes 28 structured Q&A interview scenarios.**

[Explore Chapters](#-book-architecture) · [View Progress](#-detailed-chapter-progress) · [Build the Book](#%EF%B8%8F-build-pipeline) · [Contributing](#-contributing)

</div>

---

## 🎯 What Makes This Different

Most ML interview books give you definitions to memorize. This one makes you **build everything from the ground up**, then **breaks your intuitions** with carefully designed experiments that expose where theory meets reality.

| | **This Book** | **Typical Interview Prep** |
|---|---|---|
| **Algorithms** | Implemented from scratch in NumPy, gradient-checked, output-verified against sklearn | Described conceptually with pseudocode |
| **Mathematics** | Proofs derived, then *tested* (e.g., bias-variance measured empirically, UAT width sweeps) | Formulas presented without verification |
| **Gotchas & Traps** | Deliberately demonstrated — scaling traps, leakage, dying ReLU, exposure bias, mode collapse | Mentioned in passing |
| **Q&A Quality** | 28 boxes/chapter with multi-step reasoning, code scenarios, and "why does this fail?" questions | Surface-level recall questions |
| **Code Execution** | Every listing executed and output verified — 204 listings, 0 failures | Code snippets may not run |
| **Figures** | 57 matplotlib diagrams generated *by* the code listings, not drawn by hand | Stock diagrams or none |

---

## 📊 Project at a Glance

<div align="center">

```
26 chapters drafted · 8 verified · 1.6 MB of manuscript source
730+ Q&A interview boxes · 204 executed code listings · 57 figures
700+ pages of compiled PDF · 39 planned chapters across 8 parts
```

</div>

| Metric | Count |
|---|---|
| **Total Chapters Planned** | 39 + 5 appendices |
| **Chapters Drafted** | 26 (Ch 1–26) |
| **Chapters Verified** | 8 (Ch 4–11, all code re-executed) |
| **Q&A Interview Boxes** | 730+ |
| **From-Scratch Code Listings** | 204 (all executed, 0 failures) |
| **Matplotlib Figures** | 57 |
| **Compiled PDF Pages** | ~700+ |
| **Manuscript Source** | 1.6 MB across 27 files |

---

## 🏗️ Book Architecture

The book is organized into **8 parts** that mirror the progression of a real ML career — from mathematical foundations through production systems and interview strategy.

### Part I — Foundations `Ch 1-3` · 🟢 Drafted

> *The bedrock. If you can't derive a gradient or reason about time complexity, nothing else matters.*

| Chapter | Topics | Key Implementations |
|---|---|---|
| **Ch 1: Mathematics for ML** | Linear algebra, multivariate calculus, probability, optimization | 28 Q&A, 6 code listings — gradient descent convergence, eigendecomposition, probability distributions |
| **Ch 2: Python for ML** | Performance profiling, memory management, vectorization, libraries | 28 Q&A, 8 code listings — timing benchmarks, memory-efficient patterns, NumPy broadcasting |
| **Ch 3: Data Structures & Algorithms** | Trees, graphs, heaps, ML-specific algorithms | 30 Q&A, 12 code listings — KNN, k-means, Fisher-Yates shuffle, reservoir sampling, weighted/stratified sampling from scratch |

---

### Part II — Classical Machine Learning `Ch 4-11` · ✅ Verified

> *Every algorithm built from raw NumPy, output-verified against sklearn. All 67 code listings across this part have been re-extracted and re-executed — 67 pass, 0 fail.*

| Chapter | Topics | Headline Results |
|---|---|---|
| **Ch 4: ML Fundamentals** | Bias-variance, data leakage, cross-validation | Leakage demo: 0.50 → 0.90 on pure noise; walk-forward vs shuffled: 131× optimism |
| **Ch 5: Regression** | OLS 3 ways, Lasso, Ridge, Logistic, GLMs | OLS 3 implementations agree to 8.9e-16; Lasso zeros 11/15 features; collinearity std 1.81 at ρ=0.999 |
| **Ch 6: Classification** | KNN, Naive Bayes, SVMs, Decision Trees | KNN scaling trap: 0.887 → 0.792 without normalization; CCP pruning: 90 → 11 leaves, test +4.4pt |
| **Ch 7: Ensemble Methods** | Bagging, AdaBoost, GBM, XGBoost, Stacking | Scratch AdaBoost == sklearn 0.860; stacking wins 0.868 with learned weights (incl. negative NB coef) |
| **Ch 8: Unsupervised Learning** | K-Means, DBSCAN, GMMs, PCA, t-SNE, UMAP | Scratch EM monotone log-likelihood assert; PCA/LDA/t-SNE/UMAP bake-off: 0.64 → 0.98 |
| **Ch 9: Feature Engineering** | Missing data, encoding, scaling, selection | Target-encoding leak: +8pt fiction (0.688 → 0.780) vs honest OOF 0.695; imbalance PR-AUC baseline beats all resampling |
| **Ch 10: Model Evaluation** | ROC, PR, calibration, ranking, statistical tests | McNemar χ²=58.3 p<1e-4; bootstrap AUC gap CI [+0.039, +0.055]; split conformal coverage 90% → 0.904 |
| **Ch 11: Interpretability** | Permutation, SHAP, LIME, PDP/ICE, Fairness | Scratch exact Shapley == KernelSHAP to 4 decimals; PDP hides group slopes −1.68/+1.69, ICE exposes |

---

### Part III — Deep Learning `Ch 12-18` · 🟢 Drafted

> *Full backpropagation derived and gradient-checked for every architecture. Every forward and backward pass is hand-written in NumPy with numerical gradient verification.*

| Chapter | Topics | Headline Results |
|---|---|---|
| **Ch 12: NN Fundamentals** | Perceptrons, MLPs, activations, initialization, UAT | XOR: perceptron 0.50 forever, 2-2-1 tanh MLP solves with probs 0.001/0.999; gradient-checked to 1.05e-9 |
| **Ch 13: Training** | SGD variants, Adam, schedules, BN/LN, dropout, HPO | Adam-L2 vs AdamW: 55× decoupling disparity; BN gives 20× LR headroom; Optuna TPE beats grid/random |
| **Ch 14: CNNs** | Convolutions, pooling, residual nets, transfer learning | im2col scratch CNN 0.951 on digits; residual vs plain: 0.980 vs 0.522 at 16 layers; depthwise 8.6× param reduction |
| **Ch 15: RNNs & Sequences** | Vanilla RNN, LSTM, GRU, Seq2Seq, Attention | LSTM recall: 0.973@T20 vs RNN chance; scratch Bahdanau attention backward checked to 1e-9; alignment → exact anti-diagonal |
| **Ch 16: Transformers** | Scaled dot-product, multi-head, positional encodings, KV cache | Unscaled softmax Jacobian: 8-order gradient death; capstone 2-block decoder: reversal 1.000/1.000 vs Seq2Seq 0.522/0.000 |
| **Ch 17: Computer Vision** | Object detection, segmentation, ViT, CLIP | 8 matplotlib figures; scratch U-Net skip ablation Dice 0.983 → 0.993; ViT loses to MLP = inductive-bias lesson |
| **Ch 18: Generative Models** | AE, VAE, GAN, DDPM, Normalizing Flows, CFG | Reparam vs REINFORCE variance: 4.0 vs 220 (55×); GAN mode collapse: non-saturating 7/8 modes; DDPM two-moons from scratch |

---

### Part IV — NLP & Large Language Models `Ch 19-26` · 🟢 Drafted

> *A complete pretrain → finetune → align → deploy pipeline, measured at miniature scale. Includes the full RLHF stack, speculative decoding, and from-scratch RAG retrieval.*

| Chapter | Topics | Headline Results |
|---|---|---|
| **Ch 19: Classical & Neural NLP** | Zipf's law, TF-IDF, Word2Vec, BPE, HMM, NER | Scratch SGNS with anisotropy lesson (mean cos 0.59 → centered 0.10); HMM Viterbi verified == brute force over 5⁵ paths |
| **Ch 20: Pretrained LMs** | MLM, CLM pretraining, transfer, probing, distillation | Transfer headline: 50-label pretrained 0.781 vs scratch 0.563; pretrained@50 > scratch@1000 (~20× label multiplier) |
| **Ch 21: LLM Training & Alignment** | SFT, RLHF, DPO, LoRA, Quantization, MoE | LoRA rank-4: 0.766 vs full-FT 0.764 at 1.0% params; INT8 free / INT4 per-channel 145; DPO margin 0 → +6.38 |
| **Ch 22: LLM Inference & Decoding** | Temperature, top-p, beam search, KV cache, speculative decoding | Speculative decoding UNBIASED (TV 0.079 == baseline); PagedAttention: 16% → 97.9% utilization; continuous batching +249% throughput |
| **Ch 23: Prompt Engineering** | ICL, CoT, self-consistency, constrained decoding, ReAct, injection | CoT scratchpad: ≥0.90 through T8 (direct: 0.18); prompt injection: undefended ASR 1.00, defense-in-depth 0.008 |
| **Ch 24: RAG** | Chunking, embeddings, ANN (IVF/HNSW), hybrid search, reranking | HNSW 0.969 recall at 13× fewer distance evals; hybrid RRF two-regime honest result; rerank lifts recall@10 +44% |
| **Ch 25: AI Agents** | ReAct, planning, reflection, tool-use, multi-agent, memory, eval | Agent loops, graph-based orchestration, memory architectures, evaluation frameworks |
| **Ch 26: ML System Design** | End-to-end system design patterns, scaling, serving | Design patterns for production ML systems |

---

### Part V — ML Systems & Production `Ch 27-29` · ⏳ Planned

> *System design interviews, MLOps, data engineering, and distributed training.*

| Chapter | Topics |
|---|---|
| **Ch 27: MLOps & Deployment** | CI/CD for ML, model registries, A/B testing, monitoring |
| **Ch 28: Data Engineering for ML** | Feature stores, data pipelines, data quality |
| **Ch 29: Distributed Training** | Data/model parallelism, gradient compression, fault tolerance |

---

### Part VI — Specialized Topics `Ch 30-33` · ⏳ Planned

| Chapter | Topics |
|---|---|
| **Ch 30: Recommender Systems** | Collaborative filtering, content-based, hybrid, cold start |
| **Ch 31: Time Series** | ARIMA, Prophet, temporal fusion transformers |
| **Ch 32: Reinforcement Learning** | MDPs, policy gradient, DQN, PPO, RLHF connections |
| **Ch 33: Responsible AI** | Fairness metrics, bias mitigation, privacy, safety |

---

### Part VII — Coding & Practical Rounds `Ch 34-36` · ⏳ Planned

| Chapter | Topics |
|---|---|
| **Ch 34: Implement From Scratch** | Live coding: logistic regression, decision trees, attention in 30 min |
| **Ch 35: PyTorch/TensorFlow** | Framework-specific questions, custom layers, debugging |
| **Ch 36: ML Debugging Scenarios** | Diagnosing training failures, data issues, deployment bugs |

---

### Part VIII — The Interview Itself `Ch 37-39` · ⏳ Planned

| Chapter | Topics |
|---|---|
| **Ch 37: Interview Formats & Strategy** | Phone screens, onsites, take-homes, timeline planning |
| **Ch 38: Behavioral & Projects** | STAR method, presenting ML projects, failure stories |
| **Ch 39: Case Studies & Mocks** | End-to-end mock interviews with scoring rubrics |

---

### Appendices · ⏳ Planned

| Appendix | Content |
|---|---|
| **A: Cheat Sheets** | One-page summaries for each major topic |
| **B: Glossary** | 200+ terms defined precisely |
| **C: 100 Rapid-Fire Questions** | Quick-recall warmup questions |
| **D: Datasets & Resources** | Curated links to datasets, papers, courses |
| **E: SQL & Python Reference** | Common patterns for data manipulation interviews |

---

## 🧬 How Each Chapter Is Built

Every chapter in this book follows a rigorous, repeatable methodology:

```
┌─────────────────────────────────────────────────────────────┐
│  1. THEORY          Mathematical derivations, proofs,       │
│                     and formulations with full notation      │
├─────────────────────────────────────────────────────────────┤
│  2. FROM-SCRATCH    Pure Python/NumPy implementation         │
│     CODE            with gradient checks (rel err < 1e-6)   │
├─────────────────────────────────────────────────────────────┤
│  3. VERIFICATION    Output compared against sklearn/PyTorch  │
│                     to numerical precision                   │
├─────────────────────────────────────────────────────────────┤
│  4. EXPERIMENTS     Controlled experiments that expose       │
│                     failure modes, scaling traps, and        │
│                     counter-intuitive behaviors              │
├─────────────────────────────────────────────────────────────┤
│  5. FIGURES         Matplotlib visualizations generated      │
│                     by the code listings themselves           │
├─────────────────────────────────────────────────────────────┤
│  6. Q&A BOXES       28 structured interview questions per    │
│                     chapter with detailed answers            │
└─────────────────────────────────────────────────────────────┘
```

### Verification Standard

The Part II verification pass (Chapters 4–11) set the bar:
- **67 code listings** re-extracted from manuscripts and re-executed
- **67 pass, 0 fail**
- All numerical outputs match the claims in the text
- Part II PDF: **207 pages**, spot-checked for TOC, formatting, and content accuracy

---

## 🛠️ Build Pipeline

The book uses a custom Markdown → HTML → PDF pipeline with professional typesetting:

```bash
# Build a single chapter
python AI_ML_Interview_Book/build/build_pdf.py --file manuscript/part_02_classical_ml/ch05_regression.md

# Build an entire part
python AI_ML_Interview_Book/build/build_pdf.py --part 2

# Build the complete book
python AI_ML_Interview_Book/build/build_pdf.py --full
```

### Build Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Build orchestration |
| **Markdown** | Source format for all chapters |
| **WeasyPrint** | HTML/CSS → PDF compilation with professional typography |
| **Matplotlib** | Mathematical SVG rendering (chosen over MathML after evaluation) |
| **Pygments** | Syntax highlighting for code listings |
| **Custom CSS** | Professional book styling — cover pages, Q&A boxes, code blocks, page numbers |

### Output Structure

```
AI_ML_Interview_Book/
├── build/
│   ├── build_pdf.py          # Build orchestrator
│   └── style.css             # Book typography & styling
├── manuscript/
│   ├── 00_front_matter/      # Cover, TOC
│   ├── part_01_foundations/   # Ch 1-3
│   ├── part_02_classical_ml/ # Ch 4-11
│   ├── part_03_deep_learning/# Ch 12-18
│   ├── part_04_nlp_llms/     # Ch 19-25
│   ├── part_05_ml_systems/   # Ch 26-29
│   ├── part_06_specialized/  # Ch 30-33
│   ├── part_07_coding_rounds/# Ch 34-36
│   ├── part_08_the_interview/# Ch 37-39
│   ├── appendices/           # A-E
│   ├── figures/              # 57 matplotlib-generated diagrams
│   └── PROGRESS.md           # Detailed build log
└── output/
    └── parts/                # Compiled chapter & part PDFs
```

---

## 📈 Detailed Chapter Progress

| # | Chapter | Status | Q&A | Listings | Pages |
|---|---|---|---|---|---|
| FM | Front Matter | 🟢 Drafted | — | — | — |
| 1 | Mathematics for ML | 🟢 Drafted | 28 | 6 | ~25 |
| 2 | Python for ML | 🟢 Drafted | 28 | 8 | ~28 |
| 3 | Data Structures & Algorithms | 🟢 Drafted | 30 | 12 | ~32 |
| 4 | ML Fundamentals | ✅ Verified | 28 | 8 | ~26 |
| 5 | Regression | ✅ Verified | 28 | 8 | ~28 |
| 6 | Classification | ✅ Verified | 28 | 8 | ~26 |
| 7 | Ensemble Methods | ✅ Verified | 28 | 8 | ~26 |
| 8 | Unsupervised Learning | ✅ Verified | 28 | 10 | ~29 |
| 9 | Feature Engineering | ✅ Verified | 28 | 8 | ~29 |
| 10 | Model Evaluation | ✅ Verified | 28 | 9 | ~25 |
| 11 | Interpretability | ✅ Verified | 28 | 8 | ~28 |
| 12 | NN Fundamentals | 🟢 Drafted | 28 | 8 | ~25 |
| 13 | Training Deep Networks | 🟢 Drafted | 28 | 9 | ~28 |
| 14 | CNNs | 🟢 Drafted | 28 | 6 | ~23 |
| 15 | RNNs & Sequence Models | 🟢 Drafted | 28 | 7 | ~28 |
| 16 | Transformers | 🟢 Drafted | 28 | 8 | ~28 |
| 17 | Computer Vision | 🟢 Drafted | 28 | 8 | ~32 |
| 18 | Generative Models | 🟢 Drafted | 28 | 8 | ~30 |
| 19 | Classical & Neural NLP | 🟢 Drafted | 28 | 8 | ~31 |
| 20 | Pretrained LMs | 🟢 Drafted | 28 | 8 | ~31 |
| 21 | LLM Training & Alignment | 🟢 Drafted | 28 | 8 | ~30 |
| 22 | LLM Inference & Decoding | 🟢 Drafted | 28 | 8 | ~31 |
| 23 | Prompt Engineering | 🟢 Drafted | 28 | 8 | ~33 |
| 24 | RAG | 🟢 Drafted | 28 | 8 | ~32 |
| 25 | AI Agents | 🟢 Drafted | 28 | 8 | ~30 |
| 26 | ML System Design | 🟢 Drafted | — | — | — |
| 27–39 | Remaining Chapters | ⏳ Planned | — | — | — |
| A–E | Appendices | ⏳ Planned | — | — | — |

**Legend:** ✅ Verified = all code re-extracted and re-executed with 0 failures · 🟢 Drafted = complete first draft with executed code · ⏳ Planned = outlined

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install weasyprint markdown pygments matplotlib
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/AyushDas4890/AI-Interview-Prep.git
cd AI-Interview-Prep

# Build Chapter 5 (Regression) as a standalone PDF
python AI_ML_Interview_Book/build/build_pdf.py --file manuscript/part_02_classical_ml/ch05_regression.md

# Build the entire Part II (Classical ML) — 207 pages
python AI_ML_Interview_Book/build/build_pdf.py --part 2

# Output appears in AI_ML_Interview_Book/output/parts/
```

### Reading the Manuscript

Each chapter is a self-contained Markdown file that can be read directly on GitHub or locally. The manuscript files are located at:

```
AI_ML_Interview_Book/manuscript/part_XX_<topic>/chYY_<name>.md
```

---

## 🤝 Contributing

This is an active, solo-authored project. If you find errors in code outputs, mathematical derivations, or Q&A answers, please open an issue with:

1. **Chapter and listing number** (e.g., "Ch 7, Listing 4")
2. **Expected vs. actual** behavior
3. **Your environment** (Python version, OS)

---

## 📜 License

This project is provided for educational and personal use.

---

<div align="center">

**Built from scratch. Verified by execution. Designed to make you dangerous in interviews.**

*By [Ayush Das](https://github.com/AyushDas4890)*

</div>
