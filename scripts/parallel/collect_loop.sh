#!/bin/bash
# Every 10 min: collect finished kernels; on anything new, regenerate
# plans/results-full-image.md, commit the raw results and push, so no result
# lives only in this worktree.
cd "$(dirname "$0")/../.."
while true; do
  out=$(for s in input_diag ood_full regen_queue train_queue; do python3 scripts/parallel/$s.py collect 2>&1; done)
  new=$(echo "$out" | grep -E "^\[ok\]")
  echo "$(date +%H:%M:%S) collected: $(echo -n "$new" | grep -c ok) | waiting: $(echo "$out" | grep -c '^\[wait\]')"
  if [ -n "$new" ]; then
    echo "$new"
    python3 scripts/parallel/results_summary.py
    python3 scripts/parallel/paper_tables.py > /dev/null
    find outputs/input_diag outputs/regen_full outputs/ood_full outputs/train_qfx -type f \( -name "*.json" -o -name "*.jsonl" \) \
         -not -path "*/_raw/*" -print0 | xargs -0 git add -f
    git add -f outputs/parallel/ledger.json plans/results-full-image.md plans/paper-tables-full-image.md
    git commit -q -m "results: auto-collect $(date +%H:%M) ($(echo "$new" | grep -c ok) new)" && \
      for i in 1 2 3; do git push -q && break; sleep 30; done
  fi
  sleep 600
done
