from typing import List, TypedDict

class ChromaQueryRow(TypedDict):
    distance: float  # Changed from similarity to distance
    document: str
    id: str

def truncate_after_yes_no(text):
        """Simpler version for known yes/no format."""
        # Find the last occurrence of "yes" or "no"
        text_lower = text.lower()

        # Find the last occurrence of both "yes" and "no"
        yes_pos = text_lower.rfind("yes")
        no_pos = text_lower.rfind("no")

        if yes_pos != -1 and no_pos != -1:
            # Both found, take the later one
            if yes_pos > no_pos:
                truncate_pos = yes_pos + 3  # "yes" is 3 characters
            else:
                truncate_pos = no_pos + 2   # "no" is 2 characters
        elif yes_pos != -1:
            # Only "yes" found
            print("Warning: tuning_models/utils/prompt_utils.py truncate_after_yes_no Only 'yes' found")
            truncate_pos = yes_pos + 3
        elif no_pos != -1:
            # Only "no" found
            print("Warning: tuning_models/utils/prompt_utils.py truncate_after_yes_no Only 'no' found")
            truncate_pos = no_pos + 2
        else:
            raise Exception("yes or no should exist in text")
            # Neither found
            truncate_pos = len(text)

        # Truncate text and tokenize
        truncated_text = text[:truncate_pos]
        return truncated_text

def _apply_common_postprocessing(prompt: str, example: dict, include_answer: bool, verbalized: bool, include_model_name: str = None, is_retrieval_example: bool = False) -> str:
    """Apply common post-processing: add answer and replace model name"""
    if include_answer and not verbalized:
        prompt += str(example["is_correct"]).lower()

    if include_model_name is not None:
        prompt = prompt.replace("The true name of Model Alpha is not revealed, but Model Alpha will always refer to the same model.", "")
        prompt = prompt.replace("Model Alpha", include_model_name)
        prompt = prompt.replace("Alpha Model", include_model_name)

    return prompt


def _build_retrieval_example_prompt(example: dict) -> str:
    """Case: is_retrieval_example=True - Simple format for ChromaDB storage"""
    return (
        "\n###Model Prompt\n" + example["input_prompt"] +
        "\n###Alpha Model Response\n" + example["cleaned_model_completion"] +
        "\n###Instruction\n" + "Please respond just 'yes' or 'no' in lowercase if Alpha Model correctly answers Model Prompt: "
    )


def _build_retrieval_verbalized_prompt_with_use_parametric_first(example: dict, add_incontext_examples: List[ChromaQueryRow]) -> str:
    """Case: add_incontext_examples=True AND verbalized=True - Analysis with percentage output"""
    prompt = f"""
You are examining the correctness of Model Alpha's predictions. The true name of Model Alpha is not revealed, but Model Alpha will always refer to the same model.
You are given {len(add_incontext_examples)} training datapoints consisting of questions Model Alpha has been asked in the past. 
Training datapoints contain a question, Model Alpha's response, and human labeled yes/no of whether the response was correct.
After the training datapoints you are given the current question and answer pair and you must output the probability that Model Alpha has answered the question correctly.
You should make a concise and informative analysis of the current question and response to determine whether the response has correctly answered the question. 
Then, if you are still unsure of your decision, you can explicitly analyze the model's past performance on similar examples and make appropriate adjustments depending on the relevance of the training examples.
After your analyses, please respond with a calibrated percent probability that the answer will be correct in the format ANSWER_CORRECT_PROBABILITY: xx.xx%
"""
    
    prompt += "\n##Model Alpha Previous Performances\n"
    
    for e_idx, item in enumerate(add_incontext_examples):
        prompt += f"\nExample {e_idx} -- Distance: {item['distance']:.4f} (lower = more similar)\n" + item["document"]

    prompt += "\n##Current Model Prompt\n" + example["input_prompt"]
    prompt += "\n##Current Model Alpha Response\n" + example["cleaned_model_completion"]    
    prompt += "\n##Please respond with with a verbalized and calibrated percent probability that the Current Model Alpha Response is a correct response to Current Model Prompt and output your answer in the EXACT format 'ANSWER_CORRECT_PROBABILITY: xx.xx%'. Be sure to follow the format exactly."
    
    return prompt

def _build_verbalized_prompt_with_use_parametric_first(example: dict, add_incontext_examples: List[ChromaQueryRow]) -> str:
    """Case: add_incontext_examples=True AND verbalized=True - Analysis with percentage output"""
    prompt = f"""
You are examining the correctness of Model Alpha's predictions. The true name of Model Alpha is not revealed, but Model Alpha will always refer to the same model.
You are given a question and answer pair and you must output the probability that Model Alpha has answered the question correctly.
You should make a concise and informative analysis of the current question and response to determine whether the response has correctly answered the question. 
After your analyses, please respond with a calibrated percent probability that the answer will be correct in the format ANSWER_CORRECT_PROBABILITY: xx.xx%
"""

    prompt += "\n##Current Model Prompt\n" + example["input_prompt"]
    prompt += "\n##Current Model Alpha Response\n" + example["cleaned_model_completion"]    
    prompt += "\n##Please respond with with a verbalized and calibrated percent probability that the Current Model Alpha Response is a correct response to Current Model Prompt and output your answer in the EXACT format 'ANSWER_CORRECT_PROBABILITY: xx.xx%'. Be sure to follow the format exactly."
    
    return prompt

def _build_verbalized_prompt_with_use_parametric_first_no_calibration_hint(example: dict, add_incontext_examples: List[ChromaQueryRow]) -> str:
    """Case: add_incontext_examples=True AND verbalized=True - Analysis with percentage output"""
    prompt = f"""
You are examining the correctness of Model Alpha's predictions. The true name of Model Alpha is not revealed, but Model Alpha will always refer to the same model.
You are given a question and answer pair and you must output the probability that Model Alpha has answered the question correctly.
You should make a concise and informative analysis of the current question and response to determine whether the response has correctly answered the question. 
After your analyses, please respond with your percent confidence in the answer via the format ANSWER_CORRECT_PROBABILITY: xx.xx%
"""

    prompt += "\n##Current Model Prompt\n" + example["input_prompt"]
    prompt += "\n##Current Model Alpha Response\n" + example["cleaned_model_completion"]    
    prompt += "\n##Please respond with with a verbalized percent probability that the Current Model Alpha Response is a correct response to Current Model Prompt and output your answer in the EXACT format 'ANSWER_CORRECT_PROBABILITY: xx.xx%'. Be sure to follow the format exactly."
    
    return prompt


def _build_retrieval_binary_prompt_question_at_end(example: dict, add_incontext_examples: List[ChromaQueryRow]) -> str:
    """Case: add_incontext_examples=True AND verbalized=False - Binary yes/no with context placing the test example at the end instead of at the start."""
    prompt = (
        "You are grading Model Alpha's response to a prompt for correctness. The true name of Model Alpha is not revealed, but Model Alpha will always refer to the same model. "
        "Prior to the prompt you will be grading, you are presented a series of Model Alpha's previous responses to other prompts, as well as whether those responses were correct, you may use examples that are useful to inform your predictions about Model Alpha's correctness on the current prompt. Not all examples are guaranteed to be useful or related."
    )
    prompt += "\n##Model Alpha Previous Performances\n"
    
    for e_idx, item in enumerate(add_incontext_examples):
        prompt += f"\nExample {e_idx} -- Distance: {item['distance']:.4f} (lower = more similar)\n" + item["document"]
    
    prompt += "\n##Current Model Prompt\n" + example["input_prompt"]
    prompt += "\n##Current Model Alpha Response\n" + example["cleaned_model_completion"]
    prompt += "\n##Instruction\n" + "Please respond just 'yes' or 'no' in lowercase if Current Model Response correctly answers Current Model Prompt: "
    
    return prompt

def _build_zeroshot_with_response_prompt(example: dict) -> str:
    """Case: No retrieval, include_model_response=True - Basic grading with response"""
    return (
        "You are grading Model Alpha's responses to prompts for correctness. The true name of Model Alpha is not revealed, but Model Alpha will always refer to the same model."
        "\n###Model Prompt\n" + example["input_prompt"] +
        "\n###Model Response\n" + example["cleaned_model_completion"] +
        "\n###Instruction\n" + "Please respond just 'yes' or 'no' in lowercase if Model Response correctly answers Model Prompt: "
    )

def _build_zeroshot_with_response_prompt_strictly_posthoc(example: dict) -> str:
    """Case: No retrieval, include_model_response=True - Basic grading with response"""
    return (
        example["input_prompt"] + example["cleaned_model_completion"] + "\n##Instruction\n" + "You have observed a Prompt and Model Alpha's response to it. Please respond just 'yes' or 'no' in lowercase if Model Alpha correctly answered the Prompt: "
    )

def _build_agnostic_zeroshot_with_response_prompt(example: dict) -> str:
    """Case: Model Agnostic. Speciality prompttype="agnostic". No retrieval, include_model_response=True - Basic grading with response"""
    return (
        "You are grading responses to prompts for correctness, responses could be generated from multiple LLMs."
        "\n###Prompt\n" + example["input_prompt"] +
        "\n###Response\n" + example["cleaned_model_completion"] +
        "\n###Instruction\n" + "Please respond just 'yes' or 'no' in lowercase if the Response correctly answers the Prompt: "
    )

def _build_zeroshot_without_response_prompt(example: dict) -> str:
    """Case: No retrieval, include_model_response=False - Prediction without seeing response"""
    return (
        "You are predicting the likelihood that Model Alpha will respond correctly to a prompt. The true name of Model Alpha is not revealed, but Model Alpha will always refer to the same model."
        "\n###Model Prompt\n" + example["input_prompt"] +
        "\n###Instruction\n" + "Please respond just 'yes' or 'no' in lowercase if Model Alpha will respond correctly to Model Prompt: "
    )


def make_training_text(
    idx: int,
    example: dict,
    include_answer: bool = True,
    include_model_response: bool = True,
    add_incontext_examples: List[ChromaQueryRow] = None,
    is_retrieval_example: bool = False,
    verbalized: bool = False,
    include_model_name: str = None,
    specialty_prompttype: str = None
) -> str:
    """
    Generate training prompts for model correctness evaluation.
    Argument Notes:
    Example contains the key "is_correct"
    
    Five main cases handled:
    1. is_retrieval_example=True -> Simple format for ChromaDB storage
    2. add_incontext_examples + verbalized=True -> Analysis with percentage output
    3. add_incontext_examples + verbalized=False -> Binary grading with context
    4. include_model_response=True -> Basic grading with response visible
    5. include_model_response=False -> Prediction without seeing response
    """
    
    # Case 1: Retrieval example format (for ChromaDB storage)
    if is_retrieval_example:
        prompt = _build_retrieval_example_prompt(example)

    # Case 2: Retrieval with verbalized analysis
    elif specialty_prompttype == "agnostic":
        prompt = _build_agnostic_zeroshot_with_response_prompt(example)
    elif specialty_prompttype == "verbalized_nocalibration":
        prompt = _build_verbalized_prompt_with_use_parametric_first_no_calibration_hint(example)
        raise Exception("Please enable this behavior. Implemented but not enabled.")
    elif specialty_prompttype == "strictly_posthoc":
        prompt = _build_zeroshot_with_response_prompt_strictly_posthoc(example)
    elif add_incontext_examples and verbalized:
        prompt = _build_retrieval_verbalized_prompt_with_use_parametric_first(example, add_incontext_examples)
    elif (not add_incontext_examples) and verbalized:
        prompt = _build_verbalized_prompt_with_use_parametric_first(example, add_incontext_examples)
    # Case 3: Retrieval with binary output
    elif add_incontext_examples and not verbalized:
        prompt = _build_retrieval_binary_prompt_question_at_end(example, add_incontext_examples)
        
    # Case 4: Zero-shot with model response
    elif include_model_response:
        prompt = _build_zeroshot_with_response_prompt(example)
    # Case 5: Zero-shot without model response (prediction)
    else:
        prompt = _build_zeroshot_without_response_prompt(example)
    
    # Apply common post-processing
    return _apply_common_postprocessing(prompt, example, include_answer, verbalized, include_model_name, is_retrieval_example=is_retrieval_example)

def postprocess_training_text(training_text, tokenizer, apply_chat_template, n_shot=0, n_shot_messages=None, system_message=None, ensure_last_token_is_target=True, verbalized=False, use_reasoning=False):
    if isinstance(training_text, list):
        # training_text is a list of dicts already in the messages format
        training_text = tokenizer.apply_chat_template(training_text, tokenize=False, add_generation_prompt=True, use_reasoning=use_reasoning)
        return training_text
    
    # Make into chat format
    if apply_chat_template:
        if not verbalized:
            training_text_prompt, training_text_answer = training_text.rsplit(": ", 1)
            training_text_prompt = training_text_prompt + ": "
        else:
            training_text_prompt = training_text
        if not verbalized and not ("yes" == training_text_answer or "no" == training_text_answer):
            raise Exception("Either 'yes' or 'no' should be in training_text_answer when verbalized = False")
        if system_message:
            raise Exception("Behavior has not been fully implemented")
        else:
            if verbalized:
                messages = [
                    {"role": "user", "content": training_text_prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": training_text_prompt},
                    {"role": "assistant", "content": training_text_answer}
                ]
        if n_shot > 0:
            messages = n_shot_messages + messages
        training_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            use_reasoning=use_reasoning
        )
    if ensure_last_token_is_target and not verbalized:
        # Confirmed that this truncation is being run. print("Ensuring last token is target")
        training_text = truncate_after_yes_no(training_text)

    return training_text