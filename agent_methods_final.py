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
from vllm import LLM
from vllm import SamplingParams
sampling_params_initial = SamplingParams(
    # Set the maximum number of NEW tokens to generate (e.g., 512)
    max_tokens=200, 
    # Optional: Increase temperature slightly for more detailed/diverse output
    temperature=0.3, 
    top_p=0.9,
    stop=["\n\nNote:"]
)
sampling_params_final = SamplingParams(
    # Set the maximum number of NEW tokens to generate (e.g., 512)
    max_tokens=50, 
    # Optional: Increase temperature slightly for more detailed/diverse output
    temperature=0.3, 
    top_p=0.9
)

# Add a global tokenizer variable
tokenizer = None


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
    "No HS diploma": "You have not completed a high school diploma. ",
    "Some college": "You have attended some level of college but have not completed a bachelor's degree. ",
    "HS graduate or equivalent": "You have completed a high school diploma. ",
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

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


# ============================================================================
# LLM SETUP
# ============================================================================

# Global generator variable (initialized when needed)
llm = None

def initialize_llm(temp=0, seed=42):
    """
    Initialize the LLM generator pipeline, selecting a GPU-optimized Llama 3 model if CUDA is available,
    otherwise falling back to TinyLlama on CPU.
    """
    global llm, tokenizer, sampling_params_final, sampling_params_initial

    print("Starting vLLM Engine on 1 GPU (TP=1)...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,  # Test mode: single GPU
        dtype="bfloat16", 
        max_model_len=4096,
        gpu_memory_utilization=0.85
    )

    # Initialize the tokenizer from the LLM engine
    tokenizer = llm.get_tokenizer()
    
    print("✅ Engine and Tokeniser initialized.")

    random_seed = seed

    sampling_params_initial = SamplingParams(
        max_tokens=200,
        temperature=temp,
        top_p=0.9,
        seed=random_seed,
        stop=["\n\nNote:"]
    )
    sampling_params_final = SamplingParams(
        max_tokens=50,
        temperature=temp,
        top_p=0.9,
        seed=random_seed
    )
   
    print(f"Temperature set to {str(temp)}")
    print("✅ Engine initialized.")
    print(f"✅ Model loaded successfully :) (Torch version: {torch.__version__})")
    return llm



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
    question_text = survey_dict[question]
    response_value = json_object["questions"][question]
    personal_details = demographic_info(json_object['demographics'])
    
    # Get opinion description
    opinion_num, opinion_description = get_opinion_description(response_value)

    # 1. SYSTEM PROMPT: Persona, Demographics, and Scale Definition
    system_content = (
        f"You are one of roughly 500 Americans scientifically selected to represent the entire country on September 19-22 2019.\n"
        f"You will join your fellow Americans in discussing the issues confronting the US.\n"
        f"YOUR DEMOGRAPHIC INFORMATION:\n{personal_details}\n"
        f"LIKERT SCALE:\n"
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
        f"10 = Extremely favor\n"
        f"77: No opinion, 98: Skipped, 99: Refused"
    )

    # 2. USER PROMPT: Specific Task and Input Data
    user_content = (
        f"DISCUSSION QUESTION: \"{question_text}\"\n"
        f"YOUR POSITION: {opinion_num} ({opinion_description})\n"
        f"TASK: Restate your opinion using your Likert Scale Answer then write a BRIEF argument (MAX 3 sentences)"
        f"that clearly explains WHY you hold this position based on your demographics and score.\n\n"
        f"YOUR ARGUMENT:"
    )

    # Return the list of messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return messages


def update_post_opinions_prompt(json_object, self_response, agent_responses, question):
    personal_details = demographic_info(json_object['demographics'])
    question_text = survey_dict[question]
    response_value = json_object["questions"][question]
    opinion_num, opinion_description = get_opinion_description(response_value)

    formatted_agent_responses = "\n".join([f"{i+1}. {a}" for i, a in enumerate(agent_responses)])

    # 1. SYSTEM PROMPT
    system_content = (
        f"You are one of roughly 500 Americans scientifically selected to represent the entire country on September 19-22 2019.\n"
        f"You will join your fellow Americans in discussing the issues confronting the US.\n"
        f"YOUR DEMOGRAPHIC INFORMATION:\n{personal_details}\n"
        f"LIKERT SCALE:\n"
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
        f"10 = Extremely favor"
    )

    # 2. USER PROMPT
    user_content = (
        f"DISCUSSION QUESTION: \"{question_text}\"\n\n"
        f"YOUR ORIGINAL POSITION: {opinion_num} ({opinion_description})\n"
        f"YOUR ARGUMENT: {self_response}\n\n"
        f"OTHER PARTICIPANTS' ARGUMENTS:\n{formatted_agent_responses}\n\n"
        f"TASK: After hearing these arguments, decide if your opinion has changed. You may keep your original score, "
        f"move slightly in either direction, or change significantly if the arguments strongly persuaded you."
        f"Consider whether the other participants made compelling points.\n\n"
        f"Respond with exactly two lines:\n"
        f"1. Updated Rating: [Integer 0-10]\n"
        f"2. A clear explanation."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return messages

# ============================================================================
# LLM CALLING FUNCTIONS
# ============================================================================

def call_llm_argument(messages): # Argument is now 'messages', not 'prompt'
    if llm is None:
        raise RuntimeError("LLM not initialized.")

    # Apply the Llama 3 Chat Template
    # tokenize=False returns a string formatted with <|begin_of_text|>, <|user|>, etc.
    # add_generation_prompt=True adds the final <|assistant|> tag to prompt the model to speak.
    prompt_with_template = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Generate
    generated_text = llm.generate(prompt_with_template, sampling_params_initial)[0].outputs[0].text.strip()

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

    # (Optional) Clean up logic if the model repeats the prompt (usually vllm doesn't do this, but good to be safe)
    return generated_text


def call_llm_final(messages, max_retries=5): # Argument is now 'messages'
    if llm is None:
        raise RuntimeError("LLM not initialized.")

    # Apply Template
    prompt_with_template = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    for attempt in range(max_retries):
        outputs = llm.generate(prompt_with_template, sampling_params_final)
        generated_text = outputs[0].outputs[0].text.strip()
        #print(f"LLM Output (Attempt {attempt + 1}): {generated_text}")

        # Extract numeric rating
        # how does this work with temperature 0?
        rating = extract_numeric_rating(generated_text) # Simplified extraction for this example

        if rating is not None:
            return rating
        
        print(f"⚠️ Invalid response: {generated_text}")

    return None


def extract_numeric_rating(text):
    """
    Extract a numeric rating from the 'Updated Rating' line (0–10),
    robust to non-string inputs.

    Args:
        text: The generated text from LLM (string, list, etc.)

    Returns:
        Integer rating (0–10) or None if no valid rating found
    """
    import re

    # 1) Make sure we have a string; be robust to lists and others
    if isinstance(text, list):
        # Join list elements into a single string
        text = "\n".join(str(x) for x in text)
    elif not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return None

    # 2) Prefer the explicit "Updated Rating" pattern anywhere in the text
    #    e.g. "1. Updated Rating: 7" or "Updated Rating: 10"
    m = re.search(r"Updated Rating[^0-9]*([0-9]{1,2})", text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 0 <= num <= 10:
            return num

    # 3) Fallback: look at the first line only (to avoid grabbing list numbers from later lines)
    first_line = text.splitlines()[0]
    nums = re.findall(r"\d+", first_line)
    for num_str in nums:
        num = int(num_str)
        if 0 <= num <= 10:
            return num

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