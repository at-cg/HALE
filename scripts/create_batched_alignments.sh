# #!/bin/bash
# if [ "$#" -ne 4 ]; then
# 	echo "Please place batch.py in the same directory as this script."
# 	echo "This script requires 4 arguments:"
# 	echo "1. sequencing instrument i.e. ont or pacbio"
# 	echo "2. The path to the preprocessed reads."
# 	echo "3. The path to the read ids of these reads e.g. from seqkit seq -n -i."
# 	echo "4. The number of threads to be used."
# 	echo "5. The directory to output the batches of alignments."
# 	exit
# fi


# set -e
# #set -x

# minimap2='minimap2'
# script_dir=$(dirname "$0")
# batch_script="${script_dir}/batch.py"

# seq=$1
# reads=$2
# rids=$3
# num_threads=$4
# outdir=$5

# if [ ! -d $outdir ]; then
# 	mkdir $outdir
# fi

# if($seq == 'ont')
# 	$minimap2 -K8g -cx ava-ont -k25 -w17 -e200 -r150 -m2500 -z200 -f 0.005 -t${num_threads} --dual=yes $reads $reads | $batch_script $rids - $outdir
# else
# 	$minimap2 -K8g -cx ava-ont -k25 -w17 -e200 -r150 -m2500 -z200 -f 0.005 -t${num_threads} --dual=yes $reads $reads | $batch_script $rids - $outdir


# # ava-pb	PacBio CLR all-vs-all overlap mapping (-Hk19 -Xw5 -e0 -m100).
# # ava-ont	Oxford Nanopore all-vs-all overlap mapping (-k15 -Xw5 -e0 -m100 -r2k).



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
