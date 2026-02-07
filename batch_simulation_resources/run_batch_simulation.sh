#!/bin/bash

#---------------------------------------
# SLURM DIRECTIVES (Resource Requests)
#---------------------------------------

#SBATCH --job-name=LLM_MultiAgent_Sim
#SBATCH --output=logs/LLM_MultiAgent_Sim_%j_%a.out
#SBATCH --error=logs/LLM_MultiAgent_Sim_%j_%a.err
#SBATCH --nodes=1
# The nodes must be available.
#SBATCH --nodelist=tikgpu[06,07,09]
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --array=0-3

# Set a time limit for the job
#SBATCH --time=12:00:00

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

echo "Job ID: $SLURM_ARRAY_JOB_ID, Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Processing Groups: $START_GROUP to $END_GROUP"

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
echo ""

echo "Setting up the environment..."
conda activate RDS


# Run the simulation
echo "Starting simulation..."
echo ""
python main.py \
    --start_group $START_GROUP \
    --end_group $END_GROUP \
    --gpu_id $SLURM_ARRAY_TASK_ID

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


