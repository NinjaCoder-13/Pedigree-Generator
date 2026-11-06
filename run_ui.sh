#!/usr/bin/env bash
# Start the Pedigree Dataset Generator UI in your browser.
set -e
cd "$(dirname "$0")"
ENV_DIR="$(conda env list 2>/dev/null | awk '$1=="ped_sim"{print $NF}')"
if [ -z "$ENV_DIR" ]; then
  echo "The 'ped_sim' conda environment was not found. Create it with:"
  echo "  conda env create -f ped_sim_env.yml"
  exit 1
fi
export PATH="$ENV_DIR/bin:$PATH"
exec python -m streamlit run ui/app.py --browser.gatherUsageStats false --server.address localhost "$@"
