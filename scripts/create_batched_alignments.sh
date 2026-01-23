#!/bin/bash
if [ "$#" -ne 6 ]; then
	echo "Please place batch.py in the same directory as this script."
	echo "This script requires 4 arguments:"
	echo "1. The path to the preprocessed reads."
	echo "2. The path to the read ids of these reads e.g. from seqkit seq -n -i."
	echo "3. The number of threads to be used."
	echo "4. The directory to output the batches of alignments."
	exit
fi


set -e
#set -x

minimap2='minimap2'
script_dir=$(dirname "$0")
batch_script="${script_dir}/batch.py"

reads=$1
rids=$2
num_threads=$3
outdir=$4
data_type=$5
run_number=$6


if [ ! -d $outdir ]; then
	mkdir $outdir
fi



if [[ "$data_type" == "hifi" ]]; then
    if [[ "$run_number" -eq 1 ]]; then
        minimap2 -K8g -cx ava-ont \
            -k25 -w17 -e200 -r150 \
            -m1000 -z200 -f 0.0007 \
            -t "${num_threads}" \
            --dual=yes \
            "$reads" "$reads" \
        | "$batch_script" "$rids" - "$outdir"
    else
        minimap2 -K8g -cx ava-ont \
            -k25 -w17 -e200 -r150 \
            -m1000 -z200 -f 0.002 \
            -t "${num_threads}" \
            --dual=yes \
            "$reads" "$reads" \
        | "$batch_script" "$rids" - "$outdir"
    fi
elif [[ "$data_type" == "ont_simplex" ]]; then
    if [[ "$run_number" -eq 1 ]]; then
        minimap2 -K8g -cx ava-ont \
            -k25 -w17 -e200 -r250 \
            -m1000 -z200 -f 0.0001 \
            -t "${num_threads}" \
            --dual=yes \
            "$reads" "$reads" \
        | "$batch_script" "$rids" - "$outdir"
    elif [[ "$run_number" -eq 2 ]]; then
        minimap2 -K8g -cx ava-ont \
            -k25 -w17 -e200 -r250 \
            -m1000 -z200 -f 0.001 \
            -t "${num_threads}" \
            --dual=yes \
            "$reads" "$reads" \
        | "$batch_script" "$rids" - "$outdir"
    else
        minimap2 -K8g -cx ava-ont \
            -k25 -w17 -e200 -r250 \
            -m1000 -z200 -f 0.0015 \
            -t "${num_threads}" \
            --dual=yes \
            "$reads" "$reads" \
        | "$batch_script" "$rids" - "$outdir"
    fi
fi
