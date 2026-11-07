#!/usr/bin/env bash
# Start the LLM Genotype Evaluation app in your browser (http://localhost:8502).
# It uses its own conda environment, "llm_eval" (see llm_eval/environment.yml).
set -e
cd "$(dirname "$0")"
ENV_DIR="$(conda env list 2>/dev/null | awk '$1=="llm_eval"{print $NF}')"
if [ -z "$ENV_DIR" ]; then
  echo "The 'llm_eval' conda environment was not found. Create it with:"
  echo "  conda env create -f llm_eval/environment.yml"
  exit 1
fi
cd llm_eval
exec "$ENV_DIR/bin/python" -m streamlit run app.py --browser.gatherUsageStats false \
  --server.address localhost --server.port 8502 "$@"
