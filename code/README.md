# Multi-Modal Evidence Review System

This directory contains the source code for the automated insurance claims review pipeline, designed to orchestrate text and vision models to assess claims based on submitted conversations and images.

## Architecture

The pipeline uses a multi-agent orchestration pattern to split the workload intelligently:

1. **Agent 1: Claim Parser (Text)**
   Extracts structured fields (`issue_type`, `object_part`) from the raw customer chat transcript. Uses `qwen/qwen-2.5-7b-instruct` via OpenRouter.
2. **Agent 2: Image Validator (Vision)**
   Performs a quality check on the submitted images (e.g. originality, blurry flags, text injection). Uses `qwen/qwen2.5-vl-72b-instruct` via OpenRouter. It gracefully skips invalid images and flags issues early.
3. **Agent 3: Damage Assessor (Vision)**
   Uses RAG via FAISS to retrieve similar past claims and constructs a few-shot prompt. Performs a strict 2-step evaluation (Locate, then Assess) using `qwen/qwen2.5-vl-72b-instruct` to determine the definitive `claim_status` and `severity`.
4. **Agent 4: Risk & Assembly (Text)**
   Synthesizes the outputs of prior agents and historical user claims data into a final decision record, keeping the `claim_status` completely fixed. Uses `qwen/qwen-2.5-7b-instruct`.

## Configuration

Secrets are managed via `.env`.
Copy `.env.example` to `.env` and configure:
- `GEMINI_API_KEY`: Used exclusively for the `gemini-embedding-001` FAISS embedding model.
- `OPENROUTER_API_KEY`: Used for the Qwen textual and multi-modal instruction models.

## Usage

### Run the Pipeline
To run the evaluation pipeline on sample claims:
```bash
python code/main.py --input dataset/sample_claims.csv --output output_sample.csv
```

### Run Evaluation
To score the pipeline against ground-truth data:
```bash
python code/evaluation/main.py --pred output_sample.csv --truth dataset/sample_claims.csv --report evaluation_report.md
```

## Setup Notes
- RAG cache is automatically written to `rag_cache.npz` in the repository root to avoid rate-limiting the embedding API across subsequent runs.
