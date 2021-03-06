#!/bin/bash -l
#$ -l h_rt=40:00:00
#$ -P ecog-eeg
#$ -N run_all_cells


module load python3

mkdir results
mkdir results/figs
touch results/cell_fits.json
touch results/model_comparisons.json
touch results/log_likelihoods.json

first = $1
last = $2
numJobs=$((last-first))     # Count the jobs
myJobIDs=""        
                    # Initialize an empty list of job IDs
for i in `seq 0 11`; do
    jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "2")
done

for i in `seq 0 11`; do
    jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "6")
done

for i in `seq 0 11`; do
    jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "8")
done

for i in `seq 0 49`; do
    jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "28")
done

# for i in `seq 0 11`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s16")
# done

# for i in `seq 0 10`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s17")
# done

# for i in `seq 0 33`; do 
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s21")
# done

# for i in `seq 0 36`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s22")
# done

# for i in `seq 0 40`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s23")
# done

# for i in `seq 0 53`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s24")
# done

# for i in `seq 0 61`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s26")
# done

# for i in `seq 0 37`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s28")
# done

# for i in `seq 0 30`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s30")
# done

# for i in `seq 0 52`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s31")
# done

# for i in `seq 0 27`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s32")
# done

# for i in `seq 0 27`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s35")
# done

# for i in `seq 0 24`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s36")
# done

# for i in `seq 0 3`; do
#     jobID_full=$(qsub -N "cell_$i" ./run_one_cell.sh $i "s37")
# done    