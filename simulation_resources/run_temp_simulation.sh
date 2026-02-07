    #!/bin/bash

    #---------------------------------------
    # SLURM DIRECTIVES (Resource Requests)
    #---------------------------------------

    #SBATCH --job-name=LLM_SingleSim_Placeholder
    #SBATCH --output=logs/LLM_SingleSim_Placeholder_%j.out
    #SBATCH --error=logs/LLM_SingleSim_Placeholder_%j.err
    #SBATCH --nodes=1
    # The nodes must be available.
    #SBATCH --nodelist=tikgpu[06,07,09]
    #SBATCH --mem=25G
    #SBATCH --gres=gpu:1
    #SBATCH --export=ALL
    # ** REMOVED: #SBATCH --array=0-3 **

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
    echo "Multi-Agent Discussion Simulation (Single Job)"
    echo "========================================"
    echo ""

    # --- Placeholder Values for a Single Run ---
    # Define a fixed temperature for this single job
    TEMPERATURE=0.8
    # Define the full range of groups to run in this single job
    START_GROUP=1
    END_GROUP=40 
    # Note: Since this is a single job, we don't need SLURM_ARRAY_TASK_ID or complex math.

    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node: $(hostname)"
    echo "Processing Groups: $START_GROUP to $END_GROUP (All 40 groups in sequence)"
    echo "Temperature: $TEMPERATURE"


    # Check if Python is installed
    if ! command -v python3 &> /dev/null
    then
        echo "❌ Python3 could not be found. Please install Python 3."
        exit 1
    fi

    echo "✅ Python3 found: $(python3 --version)"
    echo ""

    # Check if required files exist
    if [ ! -f "main_temp.py" ]; then
        echo "❌ main_temp.py not found in current directory (Note: Changed from main.py check)"
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
    # The four runs are combined into one logical set of commands
    python main_temp.py --start_group 1 --end_group 10 --temp "$TEMPERATURE"
    python main_temp.py --start_group 11 --end_group 20 --temp "$TEMPERATURE"
    python main_temp.py --start_group 21 --end_group 30 --temp "$TEMPERATURE"
    python main_temp.py --start_group 31 --end_group 40 --temp "$TEMPERATURE"

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