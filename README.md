# HackerRank Multi-Modal Evidence Review Pipeline

A 4-agent sequential pipeline built with Gemini 2.5 Flash to automatically parse, validate, assess, and risk-score insurance claims based on user text and images.

## Setup
1. `pip install -r requirements.txt`
2. Create a `.env` file in this root directory with your API key:
   `GEMINI_API_KEY=your_key_here`

## Running the Pipeline
To run the full end-to-end pipeline:
```bash
python code/main.py --input dataset/claims.csv --output output.csv
```

## Running the Evaluation
To test the pipeline on the labeled sample set and see metrics:
```bash
python code/main.py --input dataset/sample_claims.csv --output output_sample.csv
python code/evaluation/main.py --pred output_sample.csv --truth dataset/sample_claims.csv
```

*Note: For free-tier API keys, the pipeline has built-in exponential backoff and rate-limit handling (`429` / `503` errors), which may cause processing to take longer.*
