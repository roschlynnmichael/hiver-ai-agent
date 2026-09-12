# AI Support Agent for @AmazonHelp on Twitter

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org)
[![Model: Llama 3.1:8b](https://img.shield.io/badge/Ollama-Llama_3.1_8B-orange.svg)](https://ollama.ai)
[![Embeddings: all-MiniLM-L6-v2](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Hiver SDE Intern Take-Home Assignment Submission**

This is an end-to-end, production-grade customer support AI agent for automatically triaging customer tweets based on the intent of the message, make safe critical escalations to customer support agents with relevant context and information based on certain policies used to safeguard these escalations.

## Benchmarks

These benchmarks were derived by evaluating the models on unseen test-samples. There was no train-test data leakage. 
| Model | Intent Accuracy | Intent Macro F1 | Escalation Recall | Escalation Precision | Escalation F1 | Reply ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** (Majority Class + Never Escalate) | 25.5% | 6.8% | 50.0% | 37.2% | 42.7% | 0.118 |
| **Simple Baseline** (TF-IDF + Regex + 1-NN) | 46.8% | 34.4% | 61.1% | 77.0% | 62.3% | 0.112 |
| **Our Support Agent** (Llama 3.1 8B + RAG) | **83.0%** | **81.7%** | **90.2%** | **92.7%** | **91.4%** | **0.227** |

### Key Benchmark Results:
* **Safely escalating messages that are deemed as threats (29.1% Recall)**: The simple baseline model that uses regular expression missed almost 39% of the messages. However, the LLM and RAG model that was develloped was able to catch 90.2% of the messages with a precision of 92.7%.
* **Intents were identified correctly (47.3% Macro F1)**: Regex based intent classification accuracy was 34.4% while the LLM + RAG model was able to identify intents correctly with a Macro F1 of 81.7%

### LLM as a Judge Calibration (Performed on 25 unseen test cases):
* **Adjacent Agreement**: 84.0%
* **Exact Agreement**: 48.0%
* **Quadratic Weighted Cohen's Kappa**: 0.215
* **Mean Human Rating**: 4.12/5.0
* **Mean Judge Rating**: 3.84/5.0

### System Architecture

```
     [Incoming Customer Tweet]
                         │
                         ▼
           ┌───────────────────────────┐
           │     STAGE 1: TRIAGE       │
           │ Joint Intent & Escalation │
           │     (Llama 3.1: 8B)       │
           └─────────────┬─────────────┘
                         ▼
          Is Escalation Flagged?
          ├── YES ──► Formulate immediate empathetic human handoff
          │           (No false promises, instructs DM, states reason)
          │
          └── NO  ──► ┌───────────────────────────────────┐
                      │    STAGE 2: HISTORICAL RAG        │
                      │ Vector search top-2 resolutions   │
                      │   (all-MiniLM-L6-v2 embeddings)   │
                      └─────────────────┬─────────────────┘
                                        ▼
                      ┌───────────────────────────────────┐
                      │     STAGE 3: GROUNDED GENERATION  │
                      │ Policy-constrained Twitter reply  │
                      │       (<200 chars, DM for PII)    │
                      └───────────────────────────────────┘
```

### Pre-requisites and environment setup

* **Install Python 3.14 from Microsoft Store or Python Website**
* **Install Ollama from https://ollama.com/**
* **Once Ollama is installed, run ollama pull llama3.1:8b in terminal to pull the latest version of the model**
* **Clone the repository and change directory to the repository**
* **Create and activate a python virtual environment**
* **Once installed, run pip install -r requirements.txt from project root to install the necessary dependencies**
* **You can save a copy of the raw data in the `data/raw` folder. Then run jupyter notebooks 01 and 02 to clean and preprocess the data and have a golden set samples for evaluation.**
* **Once you have the processed data ready, you can run `python evaluate_baseline.py` to get a baseline score and `python evaluate_judge_agreement.py` to start LLM as a Judge to get judge scores and ratings**
* **There is also a `test_agent.py` file that runs two tests cases against the agent to see if everything is working correctly or not**

### Reasons for using Ollama and not any other cloud API like OpenAI or GeminiAPI or ClaudeAPI
1. Locally run model does not require any internet connection
2. Cost effective
3. Privacy
4. Ability to fine-tune models on datasets

### Attributions and Citations
1. **Dataset**: Kagggle Cusstomer Support on Twitter by thoughtvector
2. **Embeddings**: Sentence transformers library using model all-MiniLM-L6-v2
3. **LLM Engine**: Meta's Llama3.1:8B served locally using Ollama
4. **Evaluation**: scikit-learn for classification and metrics, Cohen's Kappa, Rouge Score and scipy for Spearman Rank correlation