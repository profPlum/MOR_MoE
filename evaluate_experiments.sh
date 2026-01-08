#!/bin/bash

evaluate_experiment() {
    experiment_name="$1"
    echo "Evaluating $experiment_name"
    papermill JHTDB_operator.ipynb "./notebook_runs/$experiment_name.ipynb" \
     -p checkpoint_path "./lightning_logs/$experiment_name/*/last.ckpt"
}

ls ./lightning_logs/.
echo how many should we skip \(default=0\)?
read SKIP
echo how many should be evaluated?
read N

[[ $SKIP ]] || SKIP=0

\ls -t ./lightning_logs | \head -n $((N+SKIP)) | \tail -n $N | \
while read experiment_name; do
    evaluate_experiment "$experiment_name"
done
