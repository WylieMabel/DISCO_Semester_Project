# Multi-Agent Discussion Simulation

This project simulates multi-agent discussions on policy questions using Large Language Models (LLMs).

## Files

- **agent_methods.py**: Contains all required methods, dictionaries, and the ParticipantAgent class
  - Dictionaries for mapping Likert scales, demographics, etc.
  - Helper functions for generating prompts
  - LLM calling functions
  - ParticipantAgent class

- **main.py**: Main simulation script that runs discussions for all agents and questions
  - Loads participant data
  - Runs discussion simulations
  - Saves results to JSON

- **run_simulation.sh**: Bash script to run the simulation
  - Checks for required files
  - Runs the Python simulation
  - Reports success/failure

## Required Data Files

Make sure these files are in the same directory:

1. `participant_attitudes.json` - Participant data with demographics and initial opinions
2. `just_questions_REWORDED.csv` - Question mappings for opinion generation
3. `just_questions.csv` - Survey question text

## Installation

Install required Python packages:

```bash
pip install numpy pandas torch transformers
```

## Usage

### Option 1: Using the Bash Script (Recommended)

Make the script executable and run it:

```bash
chmod +x run_simulation.sh
./run_simulation.sh
```

### Option 2: Running Python Directly

```bash
python3 main.py
```

### Option 3: Customizing the Simulation

Edit `main.py` to customize:

- Number of agents participating (default: 10)
- Questions to discuss (default: ["Q2A"])
- Model to use (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Max retries for LLM calls (default: 5)

Example modification in `main.py`:

```python
# Discuss multiple questions
questions_to_discuss = ["Q2A", "Q3", "Q4", "Q5"]

# Use more agents
agent_arguments, final_rankings = run_discussion(
    participants, 
    question, 
    num_agents=20  # Changed from 10 to 20
)
```

## Output

The simulation generates a file called `simulation_results.json` containing:

- Agent arguments for each question
- Updated post-discussion opinions for each agent

## Architecture

### agent_methods.py Structure

1. **Dictionaries**
   - `likert_scale_map_text`: Maps numeric values to text descriptions
   - `gender_map_text`, `party_map_text`, `race_map_text`, `education_map_text`: Demographic mappings

2. **Helper Functions**
   - `initial_opinions()`: Generates text representation of all opinions
   - `demographic_info()`: Generates demographic description
   - `generate_argument_prompt()`: Creates prompt for argument generation
   - `update_post_opinions_prompt()`: Creates prompt for opinion update

3. **LLM Functions**
   - `initialize_llm()`: Loads the language model
   - `call_llm_argument()`: Generates arguments
   - `call_llm_final()`: Generates updated opinions

4. **ParticipantAgent Class**
   - Represents individual participants
   - Methods for generating arguments and updating opinions

### main.py Structure

1. **load_participants()**: Loads participant data from JSON
2. **run_discussion()**: Orchestrates the discussion simulation
3. **main()**: Entry point that coordinates the entire simulation

## Notes

- The default model (TinyLlama) runs on CPU
- Larger models may require GPU and more memory
- Adjust `max_new_tokens` and `temperature` in LLM calls to control output length and creativity
- The simulation includes retry logic for failed LLM calls
