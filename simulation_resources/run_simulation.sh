#!/bin/bash

# run_simulation.sh
# Bash script to run the multi-agent discussion simulation

echo "========================================"
echo "Multi-Agent Discussion Simulation"
echo "========================================"
echo ""

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
python -m venv .venv
source .venv/bin/activate

# Install required packages (optional - uncomment if needed)
echo "Installing required packages..."
pip install -r requirements.txt
# pip3 install numpy pandas torch transformers
# echo ""

# Run the simulation
echo "Starting simulation..."
echo ""
python3 main.py

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
