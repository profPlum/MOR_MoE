#!/bin/bash

EXTRA_NB_ARGS=("$@") # extra NB args saves and restores the script's "$@"

evaluate_experiment() {
    experiment_name="$1"
    version="${2:-*}"
    out_name="$experiment_name${2:+_$2}"
    echo "Evaluating $out_name"
    papermill model_eval.ipynb "./notebook_runs/$out_name.ipynb" \
     -p checkpoint_path "./lightning_logs/$experiment_name/$version/*.ckpt" \
     -p n_sim_pred_samples 10 -p n_flow_through_times 10 "${EXTRA_NB_ARGS[@]}"
    # ^ extra NB args works like "$@" would outside the function

    rm "./notebook_runs/failures/$out_name.ipynb" 2> /dev/null
    # they are somtimes moved here as post-processing step so we erase the old version
}

\ls -lthr ./lightning_logs/. | \tail -n +2 | cat -n # ls and enumermate
echo how many should we skip \(default=0\)?
read SKIP
echo how many should be evaluated?
read N
echo how many hours should we wait \(default=0\)?
read HOURS
echo Evaluate all versions? [y/N]
read ALL_VERSIONS

[[ $ALL_VERSIONS == y ]] && echo "Evaluating all versions" || echo "Evaluating only the latest version"
[[ $SKIP ]] || SKIP=0
[[ $HOURS ]] || HOURS=0

echo using extra eval notebook args "(passed as script args)": "${EXTRA_NB_ARGS[@]}"

# pre-save while the files are fixed/known
experiment_names=$(\ls -t ./lightning_logs | \tail -n +$((SKIP+1)) | \head -n $N)

sleep $((HOURS*3600))

while read experiment_name; do
    if [[ $ALL_VERSIONS == y ]]; then
        for version in $(\ls ./lightning_logs/"$experiment_name"); do
            evaluate_experiment "$experiment_name" "$version"
        done
    else
        evaluate_experiment "$experiment_name"
    fi
done <<< "$experiment_names"
