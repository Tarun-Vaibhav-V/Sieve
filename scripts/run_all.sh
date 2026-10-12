#!/bin/zsh
# Full standard-tier benchmark, one process per condition (resumable; fresh memory each pass).
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore COLUMNS=200
RUN=${RUN:-standard_main2}
( while true; do echo "$(date +%T) swap=$(sysctl -n vm.swapusage | awk '{print $6}') free=$(memory_pressure | tail -1 | awk '{print $5}')" >> runs/$RUN.mem.log; sleep 60; done ) &
MEMPID=$!
for C in baseline sieve_full truncate uniform_lingua sieve_rules sieve_nomemo reference_bigctx \
         sieve_full+noverifier sieve_full+noprefetch sieve_full+noledger sieve_full+clf_heuristic sieve_full+core6; do
  echo "=== $C $(date +%T)" >> runs/$RUN.log
  .venv/bin/python -m sieve bench --tier standard --conditions "$C" --repeats 1 --run-id $RUN --resume >> runs/$RUN.log 2>&1
done
.venv/bin/python -m sieve report --run $RUN > runs/$RUN.report.txt 2>&1
.venv/bin/python -m sieve breakeven > runs/breakeven.log 2>&1
kill $MEMPID
echo ALLDONE >> runs/$RUN.log
