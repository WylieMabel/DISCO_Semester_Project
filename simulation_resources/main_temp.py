"""
Main simulation script for multi-agent discussion.
This script runs the simulation for all agents and all questions.
"""

import json
from agent_methods import (
    initialize_llm,
    generate_argument_prompt,
    update_post_opinions_prompt,
    call_llm_argument,
    call_llm_final,
    survey_dict
)
import argparse


def load_participants(filepath='participant_attitudes.json'):
    """
    Load participant data from JSON file.

    Args:
        filepath: Path to the participant JSON file

    Returns:
        List of participant dictionaries
    """
    with open(filepath, 'r') as f:
        participant_json = json.load(f)
    return participant_json


def run_discussion(participants, question, groupnumber=1):
    """
    Run a discussion simulation for a given question with specified agents.

    Args:
        participants: List of participant JSON objects
        question: Question ID to discuss
        num_agents: Number of agents to include in discussion

    Returns:
        Tuple of (agent_arguments dict, final_rankings dict)
    """
    #print(f"\n{'='*80}")
    print(f"DISCUSSION ON QUESTION: {question}")
    #print(f"Question Text: {survey_dict[question]}")
    #print(f"{'='*80}\n")

    # Select the first num_agents participants
    # filter the participants json to only include those with group number equal to groupnumber

    filtered_agents = [p for p in participants if p['group'] == groupnumber]
    selected_agents = filtered_agents

    # Generate arguments for each agent
    #print(f"--- GENERATING ARGUMENTS ---\n")
    agent_arguments = {}

    for agent in selected_agents:
        prompt = generate_argument_prompt(agent, question)
        argument = call_llm_argument(prompt)

        agent_arguments[agent['id']] = argument

        original_opinion = agent['questions'][question]
        #print(f"Participant {agent['id']}:")
        #print(f"  Original opinion: {original_opinion}")
        #print(f"  Argument: {argument}\n")

    # Update opinions based on discussion
    #print(f"\n--- UPDATING POST-DISCUSSION OPINIONS ---\n")
    final_rankings = {}

    for agent in selected_agents:
        # Get all other agents' arguments
        other_arguments = [v for k, v in agent_arguments.items() if k != agent['id']]

        # Generate prompt for updating opinion
        prompt = update_post_opinions_prompt(
            agent, 
            agent_arguments[agent['id']], 
            other_arguments, 
            question
        )

        # Call LLM to get updated opinion (with built-in validation and retries)
        final_ranking = call_llm_final(prompt, max_retries=5)

        final_rankings[agent['id']] = final_ranking

        original_opinion = agent['questions'][question]

        #if final_ranking is not None:
            #print(f"Participant {agent['id']}:")
            #print(f"  Original opinion: {original_opinion}")
            #print(f"  Updated rating: {final_ranking}\n")
        #else:
            #print(f"Participant {agent['id']}:")
            #print(f"  Original opinion: {original_opinion}")
            #print(f"  Updated rating: FAILED TO GET VALID RESPONSE\n")

    return agent_arguments, final_rankings


def main():
    """
    Main function to run the multi-agent simulation.
    Adapted for Slurm Array Parallelism.
    """
    # 1. SETUP ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="Run Multi-Agent Simulation")
    parser.add_argument("--start_group", type=int, default=1, help="Group number to start with")
    parser.add_argument("--end_group", type=int, default=1, help="Group number to end with (inclusive)")
    parser.add_argument("--temp", type=float, default=0.3, help="Temp param for LLM")
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"MULTI-AGENT DISCUSSION SIMULATION")
    print(f"Processing Groups: {args.start_group} to {args.end_group}")
    print("="*80 + "\n")

    # Initialize LLM
    print("Initializing LLM...")
    initialize_llm(args.temp)

    # Load participants
    print("\nLoading participant data...")
    participants = load_participants('participant_attitudes.json')
    print(f"Loaded {len(participants)} participants\n")

    questions = survey_dict.keys()
    questions_to_discuss = questions 

    all_results = {}

    # 2. LOOP THROUGH DYNAMIC RANGE
    for group in range(args.start_group, args.end_group + 1):
        print(f"\n{'#'*80}")
        print(f"STARTING DISCUSSIONS FOR GROUP {group}")
        print(f"{'#'*80}\n")

        for question in questions_to_discuss:
            try:
                agent_arguments, final_rankings = run_discussion(
                    participants, 
                    question, 
                    groupnumber=group
                )

                all_results[f'Group_{group}_{question}'] = {
                    'arguments': agent_arguments,
                    'rankings': final_rankings
                }
            except Exception as e:
                print(f"⚠️ ERROR in Group {group}, Question {question}: {e}")
                # Continue to next question/group rather than crashing entirely

    # 3. SAVE TO UNIQUE FILE NAME
    # This ensures GPU 0 doesn't overwrite GPU 1's work
    temperature_string = str(args.temp).replace('.', '_')
    output_filename = f'full_run_4/simulation_results_groups_{args.start_group}_{args.end_group}_{temperature_string}.json'

    print(f"\n{'='*80}")
    print(f"SAVING RESULTS TO {output_filename}")
    print(f"{'='*80}\n")

    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Results saved to {output_filename}")
    print("\n🎉 Simulation complete for this batch!\n")


if __name__ == "__main__":
    main()