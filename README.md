# README

## Released scope

This artifact releases the DeepSeek-V3.2 experiments used for:

- RQ1: DCMI, RAG-MIA, and SIGMA on NFCorpus, SciFact, and TREC-COVID with BGE and BM25.
- RQ2: DCMI, RAG-MIA, and SIGMA against Lakera on SciFact.
- RQ3: query-side exposure of DCMI, RAG-MIA, and SIGMA on SciFact.

The remaining baselines and the complete implementation will be made
public after the paper is accepted.

## Environment and execution

Python 3.10 is recommended.

```bash
cd /path/to/AE
conda activate YOUR_ENV_NAME
python -m pip install -r code/requirements.txt

export DEEPSEEK_API_KEY="..."
export DEEPSEEK_API_BASE="https://your-openai-compatible-endpoint/v1"

./scripts/prepare_environment.sh
./scripts/run_all.sh
```


## Cache

We cache external LLM responses to avoid evaluation deviations caused by model
randomness. A cached response is reused if and only if the current query request
is exactly identical to the cached request, including its reconstructed prompt
and retrieval context. Otherwise, the code queries DeepSeek with the current
request and stores the new response for subsequent identical requests.
