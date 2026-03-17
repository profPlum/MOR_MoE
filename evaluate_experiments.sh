#!/bin/bash

evaluate_experiment() {
    experiment_name="$1"
    echo "Evaluating $experiment_name"
    papermill model_eval.ipynb "./notebook_runs/$experiment_name.ipynb" \
     -p checkpoint_path "./lightning_logs/$experiment_name/*/*.ckpt" \
     -p n_sim_pred_samples 10 -p n_flow_through_times 10
    rm "./notebook_runs/failures/$experiment_name.ipynb" 2> /dev/null
    # they are somtimes moved here as post-processing step so we erase the old version
}

ls ./lightning_logs/. | \tail -n +2 | cat -n # ls and enumermate
echo how many should we skip \(default=0\)?
read SKIP
echo how many should be evaluated?
read N

[[ $SKIP ]] || SKIP=0

\ls -t ./lightning_logs | \tail -n +$((SKIP+1)) | \head -n $N | \
while read experiment_name; do
    evaluate_experiment "$experiment_name"
done
