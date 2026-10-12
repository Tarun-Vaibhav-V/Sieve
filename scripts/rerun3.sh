#!/bin/zsh
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore COLUMNS=200 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=standard_main2
for C in sieve_full sieve_rules sieve_full+noprefetch; do
  echo "=== rerun $C $(date +%T)" >> runs/$RUN.log
  .venv/bin/python -m sieve bench --tier standard --conditions "$C" --repeats 1 --run-id $RUN --resume >> runs/$RUN.log 2>&1 || echo "FAILED $C" >> runs/$RUN.log
done
.venv/bin/python -m sieve report --run $RUN > runs/$RUN.report.txt 2>&1
echo RERUNDONE >> runs/$RUN.log
