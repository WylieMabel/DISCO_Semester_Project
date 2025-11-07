"""
Agent methods, dictionaries, and classes for multi-agent simulation.
This module contains all the required methods for calling LLMs, 
making prompts, dictionaries, and the ParticipantAgent class.
"""

import numpy as np
import pandas as pd
import json
import torch
from transformers import pipeline


# ============================================================================
# DICTIONARIES
# ============================================================================

likert_scale_map_text = {
    0: "extremely oppose",
    1: "very strongly oppose",
    2: "strongly oppose",
    3: "moderately oppose",
    4: "slightly oppose",
    5: "do not oppose or favor",
    6: "slightly favor",
    7: "moderately favor",
    8: "strongly favor",
    9: "very strongly favor",
    10: "extremely favor"
}

gender_map_text = {
    "Male": "You identify as male. ",
    "Female": "You identify as female. "
}

party_map_text = {
    "Republican": "You support the Republican party. ",
    "Democrat": "You support the Democratic party. ",
    "Independent": "You support an Independent candidate. "
}

race_map_text = {
    "White, non-Hispanic": "You identify as White, non-Hispanic. ",
    "Black, non-Hispanic": "You identify as Black, non-Hispanic. ",
    "Hispanic": "You identify as Hispanic. ",
    "Asian, non-Hispanic": "You identify as Asian, non-Hispanic. ",
    "2+, non-Hispanic": "You identify as multiracial, non-Hispanic. ",
    "Other, non-Hispanic": "You identify as Other, non-Hispanic. "
}

education_map_text = {
    "No HS Diploma": "You have not completed a high school diploma. ",
    "Some college": "You have attended some level of college but have not completed a bachelor's degree. ",
    "HS graduate": "You have completed a high school diploma. ",
    "BA or above": "You have completed a bachelor's degree or higher. "
}

likert_scale_map = {
    0: "Extremely oppose",
    1: "Very strongly oppose",
    2: "Strongly oppose",
    3: "Moderately oppose",
    4: "Slightly oppose",
    5: "In the middle",
    6: "Slightly favor",
    7: "Moderately favor",
    8: "Strongly favor",
    9: "Very strongly favor",
    10: "Extremely favor",
    77: "No opinion",
    98: "SKIPPED ON WEB/PAPI",
    99: "REFUSED"
}


# ============================================================================
# GLOBAL VARIABLES (loaded from CSV)
# ============================================================================

# Load questions from CSV
questions = pd.read_csv("just_questions_REWORDED.csv")
qnum = questions['Variable'].to_list()
qtext = questions['Variable Label'].to_list()
q_dict = dict(zip(qnum, qtext))

# Load survey questions from CSV
survey_question = pd.read_csv("just_questions.csv")
survey_qnum = survey_question['Variable'].to_list()
survey_qtext = survey_question['Variable Label'].to_list()
survey_dict = dict(zip(survey_qnum, survey_qtext))


# ============================================================================
# LLM SETUP
# ============================================================================

# Global generator variable (initialized when needed)
generator = None

def initialize_llm():
    """
    Initialize the LLM generator pipeline, selecting a GPU-optimized Llama 3 model if CUDA is available,
    otherwise falling back to TinyLlama on CPU.
    """
    global generator

    if torch.cuda.is_available():
        # GPU is available
        model_name = "meta-llama/Meta-Llama-3-8B"  # Example: Llama 2 13B (replace with actual Llama 3 model if available)
        device = 0  # Use GPU device
        print("CUDA detected. Loading Llama 3 8B model on GPU...")
    else:
        # No GPU, fallback to TinyLlama on CPU
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        device = "cpu"
        print("No CUDA detected. Loading TinyLlama on CPU...")

    generator = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device=device
    )

    print(f"✅ Model loaded successfully :) (Torch version: {torch.__version__})")
    return generator



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initial_opinions(question_responses):
    """
    Generate text representation of initial opinions on all questions.

    Args:
        question_responses: Dictionary mapping question IDs to response values

    Returns:
        String describing all opinions
    """
    text = ""
    for q in qnum:
        response = question_responses[q]
        if response <= 10 and response >= 0:
            text = text + f"You {likert_scale_map_text[response]} {q_dict[q]}\n"
    return text


def demographic_info(demographic_info_dict):
    """
    Generate text representation of demographic information.

    Args:
        demographic_info_dict: Dictionary containing demographic information

    Returns:
        String describing demographics
    """
    text = (
        f"You are {demographic_info_dict['AGE']} years old. "
        f"{gender_map_text[demographic_info_dict['GENDER']]}"
        f"{race_map_text[demographic_info_dict['RACE']]}"
        f"{education_map_text[demographic_info_dict['EDUCATION']]}"
        f"{party_map_text[demographic_info_dict['PARTYBEFORE']]}"
    )
    return text


def get_opinion_description(response_value):
    """
    Get the description of an opinion value with explanation.

    Args:
        response_value: Numeric response value

    Returns:
        Tuple of (numeric_value, description_text)
    """
    if response_value <= 10 and response_value >= 0:
        description = f"{response_value} - {likert_scale_map_text[response_value]}"
    elif response_value == 77:
        description = f"{response_value} - No opinion"
    elif response_value == 98:
        description = f"{response_value} - Skipped"
    else:
        description = f"{response_value} - Refused to answer"

    return response_value, description


def generate_argument_prompt(json_object, question):
    """
    Generate a prompt for the agent to create an argument on a question.
    Includes the original opinion prominently to ensure arguments reflect actual stance.

    Args:
        json_object: Participant JSON object
        question: Question ID

    Returns:
        Formatted prompt string
    """
    question_text = survey_dict[question]
    response_value = json_object["questions"][question]
    opinions = initial_opinions(json_object['questions'])
    personal_details = demographic_info(json_object['demographics'])

    # Get opinion description with mapping
    opinion_num, opinion_description = get_opinion_description(response_value)

    # Clarify what the opinion scale means
    if opinion_num < 5:
        stance_meaning = "you OPPOSE/DISAGREE with this policy"
    elif opinion_num == 5:
        stance_meaning = "you are NEUTRAL on this policy"
    else:
        stance_meaning = "you FAVOR/AGREE WITH this policy"

    prompt = (
        f"You are a participant in a discussion group on US policies.\n\n"
        f"YOUR DEMOGRAPHIC INFORMATION:\n"
        f"{personal_details}\n\n"
        #f"YOUR OVERALL OPINIONS ON VARIOUS POLICIES:\n"
        #f"{opinions}\n"
        f"SPECIFIC QUESTION FOR THIS DISCUSSION:\n"
        f"\"{question_text}\"\n\n"
        f"YOUR POSITION ON THIS SPECIFIC QUESTION:\n"
        f"Opinion Score: {opinion_num}\n"
        f"Meaning: {opinion_description}\n"
        f"In other words: {stance_meaning}\n\n"
        f"TASK:\n"
        f"Restate your opinion then write a BRIEF argument (MAX 3 sentences) that clearly explains WHY you hold this position. "
        f"Your argument MUST be consistent with your score of {opinion_num} ({opinion_description}). Make your argument authentic and directly relate it to your position."
        f"YOUR ARGUMENT:"
    )
    #print(prompt)
    return prompt


def update_post_opinions_prompt(json_object, self_response, agent_responses, question):
    """
    Generate a prompt for the agent to update their opinion after discussion.

    Args:
        json_object: Participant JSON object
        self_response: The agent's own argument
        agent_responses: List of other agents' arguments
        question: Question ID

    Returns:
        Formatted prompt string
    """
    opinions = initial_opinions(json_object['questions'])
    personal_details = demographic_info(json_object['demographics'])
    question_text = survey_dict[question]
    response_value = json_object["questions"][question]

    # Get original opinion description
    opinion_num, opinion_description = get_opinion_description(response_value)

    formatted_agent_responses = "\n".join([f"{i+1}. {a}" for i, a in enumerate(agent_responses)])

    prompt = (
        f"You are a participant in a discussion group on US policies.\n\n"
        f"YOUR DEMOGRAPHIC INFORMATION:\n"
        f"{personal_details}\n\n"
        #f"YOUR ORIGINAL OPINIONS:\n"
        #f"{opinions}\n"
        f"DISCUSSION QUESTION:\n"
        f"\"{question_text}\"\n\n"
        f"YOUR ORIGINAL POSITION:\n"
        f"Opinion Score: {opinion_num}\n"
        f"Meaning: {opinion_description}\n\n"
        f"YOUR ARGUMENT IN THE DISCUSSION:\n"
        f"{self_response}\n\n"
        f"OTHER PARTICIPANTS' ARGUMENTS:\n"
        f"{formatted_agent_responses}\n\n"
        f"TASK:\n"
        f"After hearing these arguments, decide if your opinion has changed. You may keep your original score, "
        f"move slightly in either direction, or change significantly if the arguments strongly persuaded you. "
        f"Consider whether the other participants made compelling points.\n\n"
        f"SCALE:\n"
        f"0 = Extremely oppose\n"
        f"1 = Very strongly oppose\n"
        f"2 = Strongly oppose\n"
        f"3 = Moderately oppose\n"
        f"4 = Slightly oppose\n"
        f"5 = Neutral / Do not oppose or favor\n"
        f"6 = Slightly favor\n"
        f"7 = Moderately favor\n"
        f"8 = Strongly favor\n"
        f"9 = Very strongly favor\n"
        f"10 = Extremely favor\n\n"
        f"RESPOND WITH TWO SEPARATE PARTS:\n"
        f"1.  **The final opinion score, clearly labeled.**\n"
        f"2.  **A clear, detailed explanation for your final score.**\n\n"
        f"Your response must start with the exact label **\"Updated Rating: X\"** (where X is your chosen integer from 0 to 10), followed by a newline, and then your explanation.\n\n"
        f"Updated Rating:"
    )
    return prompt

# ============================================================================
# LLM CALLING FUNCTIONS
# ============================================================================

def call_llm_argument(prompt):
    """
    Call LLM to generate an argument based on the prompt.

    Args:
        prompt: The formatted prompt string

    Returns:
        Generated argument text
    """
    if generator is None:
        raise RuntimeError("LLM generator not initialized. Call initialize_llm() first.")

    output = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.6,
        pad_token_id=generator.tokenizer.eos_token_id,
        top_p=0.9,
    )

    generated_text = output[0]['generated_text'].strip()

    try:
        # Extract text after "YOUR ARGUMENT:"
        response = generated_text.split("YOUR ARGUMENT:")[1].strip()
        # Remove common end tokens
        response = response.split("\n\n")[0].strip()
        if not response:
            response = generated_text
    except IndexError:
        response = generated_text

    return response


def extract_numeric_rating(text):
    """
    Extract a numeric rating from 0-10 from the LLM output.

    Args:
        text: The generated text from LLM

    Returns:
        Integer rating (0-10) or None if no valid rating found
    """
    # Remove whitespace and extract potential numbers
    text = text.strip()
    print(text)

    # Try to find a number in the text
    import re
    numbers = re.findall(r'\d+', text)

    if numbers:
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num <= 10:
                return num

    return None


def call_llm_final(prompt, max_retries=5):
    """
    Call LLM to generate final updated opinion based on the prompt.
    Validates that the response is a number between 0-10.

    Args:
        prompt: The formatted prompt string
        max_retries: Maximum number of retries to get valid response

    Returns:
        Integer rating (0-10) or None if unable to get valid response
    """
    if generator is None:
        raise RuntimeError("LLM generator not initialized. Call initialize_llm() first.")

    for attempt in range(max_retries):
        output = generator(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.5,
            pad_token_id=generator.tokenizer.eos_token_id,
            top_p=0.9,
        )

        generated_text = output[0]['generated_text'].strip()
        print(generated_text)

        # Extract numeric rating
        rating = extract_numeric_rating(generated_text.split("Updated Rating")[-1])

        if rating is not None:
            return rating

        # If we got an invalid response, show debug info and retry
        if attempt < max_retries - 1:
            print(f"  ⚠️  Invalid response (attempt {attempt + 1}/{max_retries}): '{generated_text[:80]}'")

    return None


# ============================================================================
# PARTICIPANT AGENT CLASS
# ============================================================================

class ParticipantAgent:
    """
    Represents a participant agent in the discussion simulation.
    """

    def __init__(self, json_object):
        """
        Initialize a participant agent from JSON data.

        Args:
            json_object: Dictionary containing participant data
        """
        self.participant_id = json_object['id']
        self.demographics = json_object['demographics']
        self.personal_details = demographic_info(self.demographics)
        self.responses = json_object['questions']
        self.opinions = initial_opinions(self.responses)
        self.json_object = json_object  # Store full object for later use
        self.stances = {}
        self.post_opinions = {}

    def generate_argument(self, question):
        """
        Generate an argument for a given question.

        Args:
            question: Question ID

        Returns:
            Generated argument text
        """
        prompt = generate_argument_prompt(self.json_object, question)

        # Call LLM
        response = call_llm_argument(prompt)
        self.stances[question] = response
        return response

    def update_post_opinions(self, question, agent_responses):
        """
        Update opinions after discussion based on other agents' responses.

        Args:
            question: Question ID
            agent_responses: List of other agents' arguments

        Returns:
            Updated opinion (0-10) or None if unable to get valid response
        """
        prompt = update_post_opinions_prompt(
            self.json_object,
            self.stances[question],
            agent_responses,
            question
        )

        # Call LLM with validation
        response = call_llm_final(prompt, max_retries=5)
        self.post_opinions[question] = response
        return response

    def get_post_opinions(self):
        """
        Get all post-discussion opinions.

        Returns:
            Dictionary of post-discussion opinions
        """
        return self.post_opinions
