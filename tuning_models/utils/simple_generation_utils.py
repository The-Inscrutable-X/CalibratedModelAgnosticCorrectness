import numpy as np
from openai import OpenAI
import torch

def generate_openai(
    client: OpenAI,
    model_name: str,
    prompt: str,
    generate_kwargs: dict = None,
    chat_template: bool = True,
    verbose = False,
    system_message = None
):
    """Generation from openai compatible API"""
    if chat_template:
        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=generate_kwargs["temperature"],
            top_p=generate_kwargs["top_p"],
            max_tokens=generate_kwargs["max_new_tokens"],
            extra_body={
                "repetition_penalty": 1.05,
            }
        )
        verbose and print(chat_response)
        generated_text = chat_response.choices[0].message.content
    else:
        verbose and print(prompt)
        resp = client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=generate_kwargs["temperature"],
            top_p=generate_kwargs["top_p"],
            max_tokens=generate_kwargs["max_new_tokens"],
            extra_body={
                "repetition_penalty": 1.05,
            }
            )
        verbose and print(resp)
        generated_text = resp.choices[0].text
    return generated_text


def generate_offline(
    model, tokenizer, input_prompt, generate_kwargs=None, use_vllm=False, apply_chat_template=True,
    system_message=None, verbose=False, use_reasoning=False, sampling_params=None, using_unsloth=False,
    batched_eval=False, lora_request=None
):
    """
    Generate text using either local model or vLLM.
    generate_kwargs specify args for huggingface generate, which defaults to values set here.
    sampling_params specify args for vllm generate, which defaults to generate_kwargs if not found.
    If batched_eval is True, input_prompt should be a list of strings.
    """
    # If batched_eval, input_prompt is a list; else, it's a string
    if batched_eval:
        if not use_vllm:
            raise NotImplementedError("Batched eval is only implemented for vLLM backend.")
        # Apply chat template if requested
        prompts = input_prompt
        if apply_chat_template:
            prompts_with_template = []
            for prompt in prompts:
                if system_message:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                prompt_with_template = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=use_reasoning
                )
                prompts_with_template.append(prompt_with_template)
            prompts = prompts_with_template
        # Prepare sampling params
        if not sampling_params:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=generate_kwargs.get('temperature', 0.7),
                top_p=generate_kwargs.get('top_p', 0.8),
                repetition_penalty=generate_kwargs.get('repetition_penalty', 1.05),
                max_tokens=generate_kwargs.get('max_new_tokens', 2048)
            )
        # Generate in batch
        if using_unsloth:
            print(f"Generating with lora_request {lora_request}")
            outputs = model.fast_generate(prompts, sampling_params, lora_request=lora_request)
        else:
            outputs = model.generate(prompts, sampling_params)
        # verbose and print(f"{outputs[:5]=}")
        # Extract generated text, strip prompt if present
        responses = []
        for prompt, output in zip(prompts, outputs):
            generated_text = output.outputs[0].text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            responses.append(generated_text)
        return responses
    else:
        # Apply chat template if requested
        if apply_chat_template:
            if system_message:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_prompt}
                ]
            else:
                messages = [
                    {"role": "user", "content": input_prompt}
                ]
            
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_reasoning
            )
        
        if use_vllm:
            # Use vLLM for generation
            # Convert generate_kwargs to SamplingParams
            if True or not sampling_params:
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=generate_kwargs.get('temperature', 0.7),
                    top_p=generate_kwargs.get('top_p', 0.8),
                    repetition_penalty=generate_kwargs.get('repetition_penalty', 1.05),
                    max_tokens=generate_kwargs.get('max_new_tokens', 2048)
                )
            
            # Generate using vLLM
            verbose and print(f"{input_prompt=}")
            if using_unsloth:
                outputs = model.fast_generate([input_prompt], sampling_params, lora_request=lora_request)
            else:
                outputs = model.generate([input_prompt], sampling_params)
            verbose and print(f"{outputs=}")
            
            # Extract generated text
            response = []
            for output in outputs:
                generated_text = output.outputs[0].text
                if generated_text.startswith(input_prompt):
                    generated_text = generated_text[len(input_prompt):]
                response.append(generated_text)
            
            if len(response) > 1:
                return response
            return response[0]
        
        else:
            # Original local generation logic
            input_prompt = tokenizer(
                input_prompt,
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = input_prompt.input_ids.cuda()
            attention_mask = input_prompt.attention_mask.cuda()

            output_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
            )
            response = []
            for i in range(output_ids.shape[0]):
                response.append(
                    tokenizer.decode(
                        output_ids[i][input_ids.shape[1] :],
                        skip_special_tokens=True,
                        ignore_tokenization_space=True,
                    )
                )

            if len(response) > 1:
                return response
            return response[0]
    

def generate_one_token_huggingface(model, tokenizer, prompt, example, device, idx, indices_for_tf, calculate_confidence_from_logits, calculate_confidence_from_logits_with_softmax_first):
    tokenized_prompt = tokenizer(prompt, padding=False, add_special_tokens=True, return_tensors="pt").to(device)
    outputs = model(**tokenized_prompt, output_hidden_states=False)
    greedy_token = torch.argmax(outputs.logits[:, -1], dim=-1)
    probabilities_for_tf = calculate_confidence_from_logits(outputs.logits, indices_for_tf, device).to("cpu").detach().numpy()
    pred = {0: True, 1: False}[np.argmax(probabilities_for_tf)]

    # debug conditional print
    idx < 5 and print(
        "Model greedy decode:",
        tokenizer.convert_ids_to_tokens(greedy_token),
        "| Model T/F pred:",
        pred,
        "Pred Probs:",
        probabilities_for_tf,
        "Pred Softmax First Probs:",
        calculate_confidence_from_logits_with_softmax_first(
            outputs.logits, indices_for_tf, device
        ),
        f"{example['correct_answer']=}",
        f"{example['cleaned_model_completion']=}",
    )