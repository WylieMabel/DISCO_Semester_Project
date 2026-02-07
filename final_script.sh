#!/bin/bash

#---------------------------------------
# SLURM DIRECTIVES (Resource Requests)
#---------------------------------------

#SBATCH --job-name=LLM_MultiAgent_Sim
#SBATCH --output=logs_final/LLM_MultiAgent_Sim_%j_%a.out
#SBATCH --error=logs_final/LLM_MultiAgent_Sim_%j_%a.err
#SBATCH --nodes=1
# The nodes must be available.
#SBATCH --nodelist=tikgpu[09]
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-2

# Set a time limit for the job
#SBATCH --time=48:00:00

#---------------------------------------
# JOB COMMANDS
#---------------------------------------

cd $RDS_DIR
echo "Job running in directory: $(pwd)"

# run_simulation.sh
# Bash script to run the multi-agent discussion simulation

echo "========================================"
echo "Multi-Agent Discussion Simulation"
echo "========================================"
echo ""

GROUPS_PER_JOB=10
OFFSET=1  # Groups start at 1, not 0

# Calculate start: (ArrayID * 10) + 1
START_GROUP=$(( ($SLURM_ARRAY_TASK_ID * $GROUPS_PER_JOB) + $OFFSET ))

# Calculate end: (ArrayID + 1) * 10
END_GROUP=$(( ($SLURM_ARRAY_TASK_ID + 1) * $GROUPS_PER_JOB ))


SEED=$((SLURM_ARRAY_TASK_ID + 7))

echo "Job ID: $SLURM_ARRAY_JOB_ID, Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Processing Groups: $START_GROUP to $END_GROUP"
echo "SEED: $SEED"


# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 could not be found. Please install Python 3."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"
echo ""

# Check if required files exist
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory"
    exit 1
fi

if [ ! -f "agent_methods.py" ]; then
    echo "❌ agent_methods.py not found in current directory"
    exit 1
fi

if [ ! -f "participant_attitudes.json" ]; then
    echo "❌ participant_attitudes.json not found in current directory"
    exit 1
fi

if [ ! -f "just_questions_REWORDED.csv" ]; then
    echo "❌ just_questions_REWORDED.csv not found in current directory"
    exit 1
fi

if [ ! -f "just_questions.csv" ]; then
    echo "❌ just_questions.csv not found in current directory"
    exit 1
fi

echo "✅ All required files found"

echo "Setting up the environment..."
conda activate RDS


# Run the simulation
echo "Starting simulation..."
echo ""
#echo "Group 1 to 10"
#python main_final.py --start_group 1 --end_group 10 --temp 0 --seed "$SEED"
#echo "Group 11 to 20"
#python main_final.py --start_group 11 --end_group 20 --temp 0 --seed "$SEED"
#echo "Group 21 to 30"
#python main_final.py --start_group 21 --end_group 30 --temp 0 --seed "$SEED"
echo "Group 1 to 40"
python main_final.py --start_group 1 --end_group 40 --temp 0.4 --seed "$SEED"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Simulation completed successfully!"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "❌ Simulation failed with errors"
    echo "========================================"
    exit 1
fi