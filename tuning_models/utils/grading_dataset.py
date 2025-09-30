#!/usr/bin/env python
# grade_answers.py  –  keeps original grading logic & prints verbatim

import argparse, json, os, gc, random
import re
from accelerate.logging import PartialState
import chromadb
from chromadb.config import Settings
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, AutoModel
from tuning_models.utils.prompt_utils import ChromaQueryRow, make_training_text, postprocess_training_text
from typing import List
from tuning_models.utils.prompt_utils import truncate_after_yes_no


class GradingDataset(Dataset):
    def __init__(self, data_directory, tokenizer=None, use_train_split=True, eval_start_p=.75, train_end_p=.75, include_answer=True, \
                 verbose=True, system_message=None, apply_chat_template=True, ensure_last_token_is_target=True, n_shot=-1, exclude_subjects=None, \
                 disable_preshuffle=False, seed=42, calibration_p=-1, include_model_response=True, use_model_name_in_prompt=False, 
                 override_splits=False, group_by_question=False, specialty_prompttype=None):
        """
        disable_preshuffle: if true, will not shuffle the dataset we read in before truncating for train and test set. 
        seed: seed for data preshuffle.
        calibration_p: if >0, then the calibration set from [train_end_p to train_end_p + calibration_p] will be returned instead of the training set.
        override_splits: if true, subset 'split' metadata using percentages. 
                        if false and split metadata exists, use the existing splits.
                        if false and no split metadata exists, error.
        group_by_question: if true, group all examples with the same input_prompt together so that during optimization,
                          all responses to the same question are processed in ~ one optimization step. This ensures the model's
                          predictive power is based on parametric knowledge and past performance rather than
                          learning answers during training.
        #TODO: Make seed something specifiable in every script that instantiates GradingDataset. Currently we're setting seed to 42 by default
        """
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.ensure_last_token_is_target = ensure_last_token_is_target
        self.apply_chat_template = apply_chat_template
        self.verbose = verbose
        self.include_model_response = include_model_response
        self.include_answer = include_answer
        self.system_message = system_message
        self.n_shot = n_shot
        self.use_model_name_in_prompt = use_model_name_in_prompt
        self.global_source_model_name = None
        self.group_by_question = group_by_question
        self.specialty_prompttype = specialty_prompttype
        if n_shot > 0:
            raise Exception("N-shot naive has not been tested yet, proceed carefully.")

        # Read and Select Data
        with open(data_directory, "r") as f:
            output_info_list = [json.loads(line) for line in f]
        full_jsonl_dataset_length = len(output_info_list)

        # If the dataset does not already contain the alpha_model_name field, we should add it. Applicable to cases where the entire dataset is generated via 1 model.
        if use_model_name_in_prompt:
            model_file_path = os.path.join(os.path.dirname(data_directory), "model_used_for_dataset_generation.txt")
            if os.path.exists(model_file_path):
                with open(model_file_path, 'r') as f:
                    self.global_source_model_name = f.read().strip()
                print(f"Loaded global source model name or using collective dataset. {self.global_source_model_name}")
                for i in output_info_list:
                    i["alpha_model_name"] = self.global_source_model_name
            else:
                print(f"Could not read global source model name or using collective dataset.")

        # Check for existing split metadata, if existing metadata exists, simply split via the metadata, else split by percentages
        has_split_metadata = len(output_info_list) > 0 and "split" in output_info_list[0]
        if has_split_metadata and not (override_splits and calibration_p > 0):
            # Use existing split metadata
            if use_train_split:
                target_split = "train" if calibration_p <= 0 else "calibration"
            else:
                target_split = "test"
            output_info_list = [entry for entry in output_info_list if entry.get("split") == target_split]
            print(f"Using existing split metadata. Loaded {len(output_info_list)} entries for '{target_split}' split.")
        elif override_splits or not has_split_metadata:
            if not override_splits and not has_split_metadata:
                raise Exception("No split metadata found and override_splits=False. Either add split metadata using add_split_metadata_to_dataset() or set override_splits=True to compute splits from percentages.")
            
            if calibration_p > 0:
                output_info_list = [entry for entry in output_info_list if entry.get("split") == "calibration"]
                output_info_list = output_info_list[:int(full_jsonl_dataset_length*(calibration_p))]
            else:
                raise Exception("Using the online splitting logic is not recommended if using multiple datasets. Find this exception in codebase to reenable.")
                # Original splitting logic
                if not disable_preshuffle:
                    random.seed(seed)
                    random.shuffle(output_info_list)
                # Make n shot prompt and truncate data
                if n_shot > 0:
                    n_shot_messages = self.make_n_shot_prompt(output_info_list[:int(len(output_info_list)*train_end_p)], n=n_shot, apply_chat_template=apply_chat_template)
                    train_start = n_shot
                else:
                    train_start = 0
                if calibration_p > 0:
                    output_info_list = output_info_list[int(len(output_info_list)*(train_end_p)) : int(len(output_info_list)*(train_end_p+calibration_p))]
                elif use_train_split:
                    output_info_list = output_info_list[train_start:int(len(output_info_list)*train_end_p)]
                else:
                    output_info_list = output_info_list[int(len(output_info_list)*eval_start_p):]

        # Filter out excluded subjects
        if exclude_subjects:
            original_count = len(output_info_list)

            new_output_info_list = []
            for item in output_info_list:
                normalized_subject = str(item["subject"][0]) if isinstance(item["subject"], list) else str(item["subject"])
                if not "subject" in item:
                    raise Exception("Field subject does not exist in dataset GradingDataset")
                if not (normalized_subject in exclude_subjects):
                    new_output_info_list.append(item)
                else:
                    # print(f"{normalized_subject=}, {exclude_subjects=}")
                    pass
            output_info_list = new_output_info_list

            filtered_count = len(output_info_list)
            if verbose:
                print(f"Excluded {original_count - filtered_count} examples from subjects: {exclude_subjects}")
                print(f"Remaining examples: {filtered_count}")

        # Group by question if requested
        if group_by_question:
            output_info_list = self._group_examples_by_question(output_info_list, seed)

        # Build Prompt
        self.output_info_list = output_info_list
        self.texts = []
        for idx, example in enumerate(output_info_list):
            model_name_to_include = None
            if self.use_model_name_in_prompt:
                model_name_to_include = example["alpha_model_name"] # example.get("alpha_model_name", self.global_source_model_name)
            
            training_text: str = make_training_text(idx, example, include_answer=include_answer, include_model_response=include_model_response, include_model_name=model_name_to_include, specialty_prompttype=self.specialty_prompttype)
            
            training_text = postprocess_training_text(training_text, self.tokenizer, apply_chat_template, n_shot)
            
            # Check if training_text ends with yes or no and implement a fix if encountering one of the common issues.
            if not (training_text.endswith("yes") or training_text.endswith("no")):
                print(f"Warning: training_text does not end with yes/no at idx {idx}: '{training_text[-50:]=}'")
                if training_text.endswith("<"):
                    training_text = training_text[:-1]
                    print(f"Behavior < caught and rectified, cleaned training_text: {training_text[-50:]=}")
                else:
                    print(f"Error: unknown behavior {training_text[-50:]=}")

            self.texts.append(training_text)
            verbose and idx < 3 and print(f"Training Prompt {idx}\n", training_text)
        try:
            self.ensure_correct_tokenization()
        except Exception as e:
            print(f"Error: ensure_correct_tokenization {e}", f"{training_text=}")
        print(f"Size of initialized dataset: {len(output_info_list)}")

    def make_n_shot_prompt(self, output_info_list, n, apply_chat_template, seed=42):
        """Takes the full output_info_list and selects 5 from the train set [naive and not semantic retrieval based]"""
        random.seed(seed)
        indexes = random.sample(range(len(output_info_list)), n)
        n_shot_data = [output_info_list[i] for i in indexes]
        messages = []
        for idx, example in enumerate(n_shot_data):
            # Make into chat format and append to messages
            training_text: str = make_training_text(idx, example, include_answer=True, include_model_response=self.include_model_response, specialty_prompttype=self.specialty_prompttype)
            raise Exception(f"Chat template function postprocess_training_text has not be integrated here.")
            if apply_chat_template:
                training_text_prompt, training_text_answer = training_text.rsplit(": ", 1)
                training_text_prompt = training_text_prompt + ": "
                if not ("yes" == training_text_answer or "no" == training_text_answer):
                    raise Exception("Either 'yes' or 'no' should be in training_text_answer")
            else:
                raise Exception("Not Implemented")
            messages.append({"role": "user", "content": training_text_prompt})
            messages.append({"role": "assistant", "content": training_text_answer})
        return messages

    def _group_examples_by_question(self, output_info_list, seed):
        """
        TODO: Need to splice and remove everything before first :, since the leading instruction could be different.
        Group examples by their input_prompt so all responses to the same question
        are together in the training data. This ensures that during optimization,
        all responses to a question are processed in one step, preserving the model's
        predictive power based on parametric knowledge rather than learning answers.
        """
        from collections import defaultdict
        import random
        
        # Group examples by question
        question_groups = defaultdict(list)
        for example in output_info_list:
            if "input_prompt" not in example:
                raise Exception("Field 'input_prompt' does not exist in dataset entry")
            question_key = example["input_prompt"].split(':', 1)[-1]
            question_groups[question_key].append(example)
        
        # Shuffle the order of question groups but keep examples within each group together
        random.seed(seed)
        question_keys = list(question_groups.keys())
        random.shuffle(question_keys)
        
        # Flatten back to a single list with grouped examples
        grouped_output_info_list = []
        for question_key in question_keys:
            # Optionally shuffle examples within each question group
            examples_for_question = question_groups[question_key]
            random.shuffle(examples_for_question)  # Shuffle responses within the question
            grouped_output_info_list.extend(examples_for_question)
        
        if self.verbose:
            print(f"Grouped {len(output_info_list)} examples into {len(question_groups)} question groups")
            group_sizes = [len(examples) for examples in question_groups.values()]
            print(f"Question group sizes: min={min(group_sizes)}, max={max(group_sizes)}, avg={sum(group_sizes)/len(group_sizes):.1f}")
        
        return grouped_output_info_list

    def __len__(self):
        return len(self.texts)
    
    def to_hf_dataset(self):
        from datasets import Dataset as hfDataset
        dataset = hfDataset.from_dict({
            "text": self.texts,
        })
        return dataset
    
    def to_hf_dataset_tokenized(self):
        dataset = self.to_hf_dataset().map(self.map_tokenize, batch_size=1)
        dataset = dataset.remove_columns("text") 
        # self.verbose and 
        print(f"In GradingDataset to_hf_dataset_tokenized() {dataset[0]=}")
        return dataset

    def map_tokenize(self, x):
        tokens = self.tokenizer(x["text"]) 
        if self.ensure_last_token_is_target:
            tokens["labels"] = [-100]*(len(tokens["input_ids"])-1)+[tokens["input_ids"][-1]]
        else:
            tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    
    def shuffle(self, seed=42):
        """This shuffle only affects the order of examples, not which examples are in train vs test"""
        import random
        combined_list = list(zip(self.texts, self.output_info_list))
        random.seed(seed)
        random.shuffle(combined_list)
        self.texts, self.output_info_list = zip(*combined_list)
        self.texts = list(self.texts)
        self.output_info_list = list(self.output_info_list)
        print(f"Dataset Shuffled")

    def truncate_to_n_samples(self, n_samples):
        self.texts = self.texts[:n_samples]
        self.output_info_list = self.output_info_list[:n_samples]
        print(f"Truncated to {n_samples} samples")

    def balance_yes_no(self, seed=42):
        """
        Filters the dataset to have an equal number of 'yes' and 'no' samples.
        """
        yes_indices = []
        no_indices = []
        for idx, info in enumerate(self.output_info_list):
            # The answer is always at the end of the prompt, after the last ": "
            if isinstance(info, dict) and "is_correct" in info:
                label = str(info["is_correct"]).lower()
            if label == "yes":
                yes_indices.append(idx)
            elif label == "no":
                no_indices.append(idx)
            # else: ignore

        min_count = min(len(yes_indices), len(no_indices))
        if min_count == 0:
            print("Warning: Only one class present, cannot balance.")
            return None

        random.seed(seed)
        yes_sample = random.sample(yes_indices, min_count)
        no_sample = random.sample(no_indices, min_count)
        selected_indices = sorted(yes_sample + no_sample)

        # Create a new instance with the same tokenizer and settings, but dummy file
        new_dataset = GradingDataset.__new__(GradingDataset)
        for attr in self.__dict__:
            setattr(new_dataset, attr, getattr(self, attr))
        # Overwrite with balanced data
        new_dataset.texts = [self.texts[i] for i in selected_indices]
        new_dataset.output_info_list = [self.output_info_list[i] for i in selected_indices]
        print(f"Balanced dataset to {min_count} 'yes' and {min_count} 'no' samples (total {2*min_count})")
        return new_dataset
    
    def ensure_correct_tokenization(self):
        if self.ensure_last_token_is_target:
            for i in self.texts:
                tokenized = self.tokenizer(i).input_ids
                # print(f"{self.tokenizer.decode(tokenized[-1])=}")
                try:
                    assert (("yes" in self.tokenizer.decode(tokenized[-1])) or ("no" in self.tokenizer.decode(tokenized[-1]))) # Last token should CONTAIN yes/no
                except:
                    print(f"{self.tokenizer.decode(tokenized[-1])=}{tokenized=}{self.tokenizer.decode(tokenized)=}{i=}")
                    raise Exception()
                try:
                    assert self.tokenizer.decode(tokenized[-2]) == self.tokenizer.decode(tokenized[-2])  # Only last token should differ
                except:
                    print(f"{self.tokenizer.decode(tokenized[-2])=}")
                    raise Exception()

    def __getitem__(self, idx):
        """
        Get prompt in tokenized form for training purposes.
        """
        raise Exception("Tokenization Not Implemented Yet")
        prompt = self.texts[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="longest", truncation=False).to(self.device)
        inputs = {k:v.squeeze(0) for k, v in inputs.items()}
        return inputs  # a dict with input_ids, correct_answers, attention masks, etc
    
    @staticmethod
    def save_predictions_for_attribution(new_dataset, save_path):
        """
        Save a .jsonl file where each line is an entry from new_dataset.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            for entry in new_dataset:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved predictions for attribution to {save_path}")

    @staticmethod
    def load_predictions_for_attribution(save_path):
        predictions_for_attribution = []
        with open(save_path, "r") as f:
            for line in f:
                predictions_for_attribution.append(json.loads(line))
        return predictions_for_attribution
    
    def format_for_grpo(self):
        from datasets import Dataset as hfDataset
        dataset = hfDataset.from_dict({
            "prompt": self.texts,
            "answer": [int(i["is_correct"] == "yes") for i in self.output_info_list]
        })
        return dataset


# Chroma Functions-----------------------------------------------------------------------------
def build_collection(texts, subjects=None, alpha_model_names=None, collection_name=None, chroma_path=None, embedding_function=None):
    """Create (or open) a Chroma collection and add documents.
    
    Args:
        texts: List of text documents to add to the collection
        subjects: List of subjects for metadata (optional)
        alpha_model_names: List of model names for metadata (optional)
        collection_name: Name of the ChromaDB collection
        chroma_path: Path for persistent storage (optional, uses ephemeral if None)
        embedding_function: String identifier for embedding function ("ReasonIR" or None for default or "default")
    """
    #TODO: Make it such that we also store model name as a metadata, and make model name a default field in all our datasets.
    if chroma_path:
        client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    else:
        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    
    # Handle embedding function routing
    ef = None
    if embedding_function == "ReasonIR":
        ef = ReasonIREmbeddingFunction()
        print("Using ReasonIREmbeddingFunction for embeddings")
    elif embedding_function is None or embedding_function == "default":
        # Use ChromaDB's default embedding function
        print("Using default ChromaDB embedding function")
    else:
        raise ValueError(f"Unrecognized embedding function: {embedding_function}. Supported values: None, 'ReasonIR'")
    
    # Use embedding function in collection name
    collection_name = f"{collection_name}-{embedding_function or 'default'}"
    if ef:
        collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)
    else:
        collection = client.get_or_create_collection(name=collection_name)
    print(f"After client.get_or_create_collection, we have this many embeddings {collection.count()}")
    all_items = collection.get()
    try:
        existing_ids = set(all_items["ids"])
    except:
        existing_ids = set()
    print(f"We have this many existing IDS: {len(existing_ids)}")
    
    # Add documents in batches
    batch_size = 100
    batch_ids = []
    batch_docs = []
    batch_metadatas = []

    if subjects == None:
        subjects = ["" for i in texts]
    if alpha_model_names == None: 
        alpha_model_names = ["" for i in texts]
    
    for idx, (doc, subj, alpha_model_name) in enumerate(zip(texts, subjects, alpha_model_names)):
        if f"train_{idx}" not in existing_ids:
            batch_ids.append(f"train_{idx}")
            batch_docs.append(doc)
            batch_metadatas.append({"subject": str(subj), "alpha_model_name": str(alpha_model_name)})
        
        # Add batch when it reaches batch_size or at the end
        if len(batch_ids) >= batch_size or idx == len(texts) - 1:
            if batch_ids:  # Only add if there are items in the batch
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                )
                print(f"Added batch of {len(batch_ids)} documents to chroma (through index {idx})")
                # Clear the batch
                batch_ids = []
                batch_docs = []
                batch_metadatas = []
        
        idx % 100 == 0 and print(f"Processed {idx} documents", flush=True)
    
    return {"collection": collection, "embedding_function": ef}

from chromadb import Documents, EmbeddingFunction, Embeddings

class ReasonIREmbeddingFunction(EmbeddingFunction):
    """Reference: https://huggingface.co/Qwen/Qwen3-32B"""
    def __init__(self, max_length=32768):
        self.max_length = max_length
        self.load()
    def __call__(self, input: Documents) -> Embeddings:
        print(f"{len(input)=}")
        embeds = self.model.encode(input, instruction="", max_length=self.max_length, batch_size=1)
        print(f"{len(embeds)=}")
        return embeds
    def unload(self):
        self.model = None
    def load(self):
        # Get the current device from accelerate state
        accelerate_state = PartialState()
        device = accelerate_state.device
        self.model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype="auto", trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()
    
class GradingDatasetWithRetrieval(GradingDataset):
    def __init__(self, data_directory, chroma_source_dataset=None, k=15, chroma_path=None, collection_name="training-examples", has_no_overlap=True, existing_chroma_collection=None, verbalized=False, use_reasoning=False, embedding_function=None, existing_embedding_function_instance=None, **kwargs):
        """
        GradingDataset with Chroma-based retrieval capabilities.
        
        Args:
            data_directory: Path to the main dataset (same as parent class), the only positional arg.
            chroma_source_dataset: GradingDataset object to build Chroma database from, it should have apply_chat_template=False.
            k: Number of similar examples to retrieve
            chroma_path: Path to store the Chroma database, is only used when existing_chroma_collection is not set
            existing_chroma_collection: use existing collection instead of regenerating.
            collection_name: Name of the Chroma collection
            embedding_function: String identifier for embedding function ("ReasonIR" or None for default)
            **kwargs: Additional arguments passed to parent GradingDataset
        """
        # Initialize parent class
        super().__init__(data_directory, **kwargs)
        
        # Store retrieval parameters
        self.k = k
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.chroma_source_dataset = chroma_source_dataset
        self.has_no_overlap = has_no_overlap
        self.verbalized = verbalized
        self.use_reasoning = use_reasoning

        # Process source dataset
        texts = chroma_source_dataset.texts
        subjects = [i.get("subject", "") for i in chroma_source_dataset.output_info_list]
        alpha_model_names = [i.get("alpha_model_name", "") for i in chroma_source_dataset.output_info_list]
        #TODO: Add get model name
        
        # Build Chroma collection from training dataset
        if existing_chroma_collection:
            self.chroma_collection = existing_chroma_collection
            self.existing_embedding_function_instance = existing_embedding_function_instance
        else:
            build_collection_outputs = build_collection(texts=texts, subjects=subjects, alpha_model_names=alpha_model_names, collection_name=collection_name, chroma_path=chroma_path, embedding_function=embedding_function)
            self.chroma_collection = build_collection_outputs["collection"]
            self.existing_embedding_function_instance = build_collection_outputs["embedding_function"]

    def get_chroma_collection(self):
        return self.chroma_collection

    def get_existing_embedding_function_instance(self):
        return self.existing_embedding_function_instance

    def augment_with_retrieved_examples(self):
        # Edit self.text
        output_info_list = self.output_info_list
        system_message = self.system_message
        apply_chat_template = self.apply_chat_template
        include_model_response = self.include_model_response
        n_shot = self.n_shot
        ensure_last_token_is_target = self.ensure_last_token_is_target
        verbose = self.verbose
        if n_shot > 0:
            raise Exception("Behavior with N-shot has not been fully implemented For GradingDatasetWithRetrieval")
        self.texts = []
        for idx, example in enumerate(output_info_list):
            model_name_to_include = None
            if self.use_model_name_in_prompt:
                model_name_to_include = example["alpha_model_name"] # example.get("alpha_model_name", self.global_source_model_name) 

            #TODO: Query text does not seem to have the chat template, while the training datapoints does. 
            query_text = make_training_text(idx, example, include_answer=self.include_answer, include_model_response=include_model_response, include_model_name=model_name_to_include)
            in_context_retrievals = self.obtain_top_k_and_filter(query_text=query_text, query_idx=idx, k=self.k)
            training_text: str = make_training_text(idx, example, include_answer=self.include_answer, include_model_response=include_model_response, add_incontext_examples=in_context_retrievals, verbalized=self.verbalized, include_model_name=model_name_to_include, specialty_prompttype=self.specialty_prompttype)
            # Make into chat format
            training_text = postprocess_training_text(
                training_text=training_text,
                tokenizer=self.tokenizer,
                apply_chat_template=apply_chat_template,
                system_message=system_message,
                ensure_last_token_is_target=ensure_last_token_is_target,
                verbalized=self.verbalized,
                use_reasoning=self.use_reasoning
            )
            self.texts.append(training_text)
            verbose and idx < 3 and print(f"Augmented Training Prompt {idx}\n", training_text)
        try:
            if not self.verbalized:
                self.ensure_correct_tokenization()
        except Exception as e:
            print(f"Error: ensure_correct_tokenization {e}")

    def augment_with_retrieved_examples_batched(self):
        # Edit self.text
        output_info_list = self.output_info_list
        system_message = self.system_message
        apply_chat_template = self.apply_chat_template
        include_model_response = self.include_model_response
        n_shot = self.n_shot
        ensure_last_token_is_target = self.ensure_last_token_is_target
        verbose = self.verbose
        if n_shot > 0:
            raise Exception("Behavior with N-shot has not been fully implemented For GradingDatasetWithRetrieval")

        # Append all queries together to allow for batched processing
        all_query_texts = []
        all_query_indices = []
        for idx, example in enumerate(output_info_list):
            model_name_to_include = None
            if self.use_model_name_in_prompt:
                model_name_to_include = example["alpha_model_name"]
            query_text = make_training_text(idx, example, include_answer=self.include_answer, include_model_response=include_model_response, include_model_name=model_name_to_include, specialty_prompttype=self.specialty_prompttype)
            all_query_texts.append(query_text)
            all_query_indices.append(idx)
        # Batch retrieve all at once
        print(f"Starting batch retrieval for {len(all_query_texts)} queries...")
        all_retrieved_results = self.obtain_top_k_and_filter_batch(
            all_query_texts, 
            all_query_indices, 
            self.k,
            batch_size=100
        )

        self.texts = []
        for idx, (example, in_context_retrievals) in enumerate(zip(output_info_list, all_retrieved_results)):
            model_name_to_include = None
            if self.use_model_name_in_prompt:
                model_name_to_include = example["alpha_model_name"] # example.get("alpha_model_name", self.global_source_model_name) 

            training_text: str = make_training_text(idx, example, include_answer=self.include_answer, include_model_response=include_model_response, add_incontext_examples=in_context_retrievals, verbalized=self.verbalized, include_model_name=model_name_to_include, specialty_prompttype=self.specialty_prompttype)
            # Make into chat format
            training_text = postprocess_training_text(
                training_text=training_text,
                tokenizer=self.tokenizer,
                apply_chat_template=apply_chat_template,
                system_message=system_message,
                ensure_last_token_is_target=ensure_last_token_is_target,
                verbalized=self.verbalized,
                use_reasoning=self.use_reasoning
            )
            self.texts.append(training_text)
            verbose and idx < 3 and print(f"Augmented Training Prompt {idx}\n", training_text)
        try:
            if not self.verbalized:
                self.ensure_correct_tokenization()
        except Exception as e:
            print(f"Error: ensure_correct_tokenization {e}")

    def obtain_top_k_and_filter(self, query_text, query_idx, k) -> List[ChromaQueryRow]:
        # Early return for k=0
        if k == 0:
            return []
            
        results = self.chroma_collection.query(
                query_texts=[query_text],
                n_results=self.k + 1,
            )
        distances = results["distances"][0]  # Changed variable name from similarities to distances
        documents = results["documents"][0]
        ids = results["ids"][0]
        # print(f"{ids=}")

        if self.has_no_overlap:
            distances, documents, ids = distances[:k], documents[:k], ids[:k]
        else:
            # This condition means we're using the train dataset, and we must remove the query from the list of results. 
            result_ids = [int(re.search(r'\d+', i).group()) for i in ids] # Extract the number part of the id
            if query_idx in result_ids:
                idx_to_remove = result_ids.index(query_idx)
                distances.pop(idx_to_remove), documents.pop(idx_to_remove), ids.pop(idx_to_remove)
            else:
                distances, documents, ids = distances[:k], documents[:k], ids[:k]
        query_idx % 1000 == 0 and print(f"{query_idx+1} queries retrieved from chroma")
        rows = [
            {"distance": dist, "document": doc, "id": id_val}  # Changed from similarity to distance
            for dist, doc, id_val in zip(distances, documents, ids)
        ]
        return rows
    
    def obtain_top_k_and_filter_batch(self, query_texts: List[str], query_indices: List[int], k: int, batch_size: int = 100) -> List[List[ChromaQueryRow]]:
        """
        Batch version of obtain_top_k_and_filter for multiple queries at once.
        
        Args:
            query_texts: List of query texts
            query_indices: List of corresponding query indices
            k: Number of results per query
            batch_size: Size of batches to process (ChromaDB has limits)
        
        Returns:
            List of lists, where each inner list contains ChromaQueryRow results for one query
        """
        if k == 0:
            return [[] for _ in query_texts]
        all_results = []
        
        # Process in batches to avoid ChromaDB memory/size limits
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i + batch_size]
            batch_indices = query_indices[i:i + batch_size]
            
            # Single batch query to ChromaDB
            results = self.chroma_collection.query(
                query_texts=batch_texts,
                n_results=k + 1,  # Get extra in case we need to filter self-matches
            )
            
            # Process each query's results in the batch
            batch_results = []
            for j, (query_idx, distances, documents, ids) in enumerate(zip(  # Changed similarities to distances
                batch_indices,
                results["distances"], 
                results["documents"], 
                results["ids"]
            )):
                if self.has_no_overlap:
                    distances, documents, ids = distances[:k], documents[:k], ids[:k]
                else:
                    # Remove self-match for training data
                    result_ids = [int(re.search(r'\d+', id_val).group()) for id_val in ids]
                    if query_idx in result_ids:
                        idx_to_remove = result_ids.index(query_idx)
                        distances.pop(idx_to_remove)
                        documents.pop(idx_to_remove) 
                        ids.pop(idx_to_remove)
                    distances, documents, ids = distances[:k], documents[:k], ids[:k]
                
                rows = [
                    {"distance": dist, "document": doc, "id": id_val}  # Changed from similarity to distance
                    for dist, doc, id_val in zip(distances, documents, ids)
                ]
                batch_results.append(rows)
            
            all_results.extend(batch_results) #TODO: Should this not be append to make 2d list?
            
            if (i + batch_size) % 1000 == 0:
                print(f"Processed {min(i + batch_size, len(query_texts))} retrieval queries in batches")
        
        return all_results

def merge_graded_datasets(dataset_paths, output_path=None, include_model_metadata=True):
    """
    Merge multiple datasets created by generate_dataset_pt2.
    
    Args:
        dataset_paths: List of paths to graded dataset JSONL files
        output_path: Optional path to save merged dataset. If None, returns data in memory
        include_model_metadata: If True, read model_used_for_dataset_generation.txt files
    
    Returns:
        List of merged dataset entries (if output_path is None)
    """
    import json
    import os
    from pathlib import Path
    
    merged_data = []
    
    for dataset_path in dataset_paths:
        print(f"Processing dataset: {dataset_path}")
        
        # Read the graded dataset
        with open(dataset_path, 'r') as f:
            dataset_entries = [json.loads(line) for line in f]
        
        alpha_model_name = None
        if include_model_metadata:
            # Look for model_used_for_dataset_generation.txt in the same directory
            dataset_dir = os.path.dirname(dataset_path)
            model_file_path = os.path.join(dataset_dir, "model_used_for_dataset_generation.txt")
            
            if os.path.exists(model_file_path):
                with open(model_file_path, 'r') as f:
                    alpha_model_name = f.read().strip()
                print(f"Found source model: {alpha_model_name}")
            else:
                print(f"Warning: No model metadata found at {model_file_path}")
        
        # Add metadata to each entry
        for entry in dataset_entries:
            if include_model_metadata and alpha_model_name:
                entry["alpha_model_name"] = alpha_model_name
            entry["dataset_source_path"] = dataset_path  # Track which file this came from
            merged_data.append(entry)
    
    print(f"Merged {len(merged_data)} total entries from {len(dataset_paths)} datasets")
    
    # Print summary of source models
    if include_model_metadata:
        model_counts = {}
        for entry in merged_data:
            model = entry.get("alpha_model_name", "Unknown")
            model_counts[model] = model_counts.get(model, 0) + 1
        print("Source model distribution:")
        for model, count in model_counts.items():
            print(f"  {model}: {count} entries")
    
    # Save to file if output_path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for entry in merged_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Saved merged dataset to {output_path}")
        
        # Save dataset paths file
        datasets_merged_path = os.path.join(os.path.dirname(output_path), "datasets_merged.txt")
        with open(datasets_merged_path, 'w') as f:
            # Check if launched with shells_launcher
            shells_log_path = os.environ.get('SHELLS_LAUNCHER_LOG_PATH')
            if shells_log_path:
                f.write(f"Shells launcher log: {shells_log_path}\n")
                f.write("=" * 50 + "\n\n")
            else:
                f.write(f"Shells launcher log: not found\n")
                f.write("=" * 50 + "\n\n")
            
            for path in dataset_paths:
                f.write(path + '\n')
    
    return merged_data

def add_split_metadata_to_dataset(dataset_path, eval_start_p=0.75, train_end_p=0.7, calibration_p=0.05, seed=42, disable_preshuffle=False):
    """
    Add split metadata to a graded dataset to preserve train/test boundaries.
    
    Args:
        dataset_path: Path to the graded dataset JSONL file
        eval_start_p: Same as used in original dataset generation (default 0.75)
        train_end_p: Same as used in original dataset generation (default 0.7)
        calibration_p: Calibration set proportion (default 0.05)
        seed: Same seed used in original dataset generation (default 42)
        disable_preshuffle: Same as used in original dataset generation (default False)
    """
    import json
    import random
    import os
    
    print(f"Adding split metadata to {dataset_path}")
    
    # Read the dataset
    with open(dataset_path, 'r') as f:
        output_info_list = [json.loads(line) for line in f]
    
    original_count = len(output_info_list)
    
    # Apply the same shuffling logic as GradingDataset
    if not disable_preshuffle:
        random.seed(seed)
        random.shuffle(output_info_list)
    
    # Apply the same splitting logic as GradingDataset
    total_len = len(output_info_list)
    train_end_idx = int(total_len * train_end_p)
    calib_end_idx = int(total_len * (train_end_p + calibration_p))
    eval_start_idx = int(total_len * eval_start_p)
    
    # Assign split labels
    for idx, entry in enumerate(output_info_list):
        if idx < train_end_idx:
            entry["split"] = "train"
        elif idx < calib_end_idx:
            entry["split"] = "calibration"
        elif idx >= eval_start_idx:
            entry["split"] = "test"
        else:
            # This handles the gap between calibration and test sets
            entry["split"] = "unused"
    
    # Count splits
    split_counts = {}
    for entry in output_info_list:
        split = entry["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print(f"Split distribution for {os.path.basename(dataset_path)}:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} entries ({count/original_count:.2%})")
    
    # Write back to the same file with split metadata
    with open(dataset_path, 'w') as f:
        for entry in output_info_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Added split metadata to {dataset_path}")
