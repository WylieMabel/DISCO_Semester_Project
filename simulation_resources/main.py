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


def run_discussion(participants, question, num_agents=10):
    """
    Run a discussion simulation for a given question with specified agents.

    Args:
        participants: List of participant JSON objects
        question: Question ID to discuss
        num_agents: Number of agents to include in discussion

    Returns:
        Tuple of (agent_arguments dict, final_rankings dict)
    """
    print(f"\n{'='*80}")
    print(f"DISCUSSION ON QUESTION: {question}")
    print(f"Question Text: {survey_dict[question]}")
    print(f"{'='*80}\n")

    # Select the first num_agents participants
    selected_agents = participants[:num_agents]

    # Generate arguments for each agent
    print(f"--- GENERATING ARGUMENTS ---\n")
    agent_arguments = {}

    for agent in selected_agents:
        prompt = generate_argument_prompt(agent, question)
        argument = call_llm_argument(prompt)

        agent_arguments[agent['id']] = argument

        original_opinion = agent['questions'][question]
        print(f"Participant {agent['id']}:")
        print(f"  Original opinion: {original_opinion}")
        print(f"  Argument: {argument}\n")

    # Update opinions based on discussion
    print(f"\n--- UPDATING POST-DISCUSSION OPINIONS ---\n")
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

        if final_ranking is not None:
            print(f"Participant {agent['id']}:")
            print(f"  Original opinion: {original_opinion}")
            print(f"  Updated rating: {final_ranking}\n")
        else:
            print(f"Participant {agent['id']}:")
            print(f"  Original opinion: {original_opinion}")
            print(f"  Updated rating: FAILED TO GET VALID RESPONSE\n")

    return agent_arguments, final_rankings


def main():
    """
    Main function to run the multi-agent simulation.
    """
    print("\n" + "="*80)
    print("MULTI-AGENT DISCUSSION SIMULATION")
    print("="*80 + "\n")

    # Initialize LLM
    print("Initializing LLM...")
    initialize_llm()

    # Load participants
    print("\nLoading participant data...")
    participants = load_participants('participant_attitudes.json')
    print(f"Loaded {len(participants)} participants\n")

    # Define questions to discuss
    # You can modify this list to include all questions you want to simulate
    questions_to_discuss = ["Q1","Q2A"]  # Add more questions as needed: ["Q2A", "Q3", "Q4", ...]

    # Run discussion for each question
    all_results = {}

    for question in questions_to_discuss:
        agent_arguments, final_rankings = run_discussion(
            participants, 
            question, 
            num_agents=4
        )

        all_results[question] = {
            'arguments': agent_arguments,
            'rankings': final_rankings
        }

    # Save results to JSON file
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    with open('simulation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("✅ Results saved to simulation_results.json")
    print("\n🎉 Simulation complete!\n")


if __name__ == "__main__":
    main()
