import traceback
from dynamics_project.tuning_models.utils.grading_dataset import GradingDataset, GradingDatasetWithRetrieval
import os
import pickle
import copy
from accelerate import PartialState

def load_dataset(dataset_name, tokenizer, ntrain=0, eval_start_p=.75, train_end_p=.75, datafile=None, shuffle=False, train_on_all_tokens=False, exclude_from_train=None, calibration_p=-1, include_model_response: bool = True, chroma_path=None, collection_name="training-examples", k_retrieve=15, chroma_source_datafile=None, verbalized=False, use_reasoning=False, use_model_name_in_prompt=False, dataset_save_path=None, dataset_load_path=None, embedding_function=None, group_by_question=False, specialty_prompttype=None, override_splits=False):
    """Return a HF dataset. Notes: chroma_source_datafile is a special argument only if you want to use a different dataset for chroma than the train dataset."""
    
    # If load path is specified, load from there
    if dataset_load_path:
        raise Exception("Not Implemented")

    # Create datasets from scratch
    if dataset_name == "CORRECTNESS_TUNE":
        train_dataset = GradingDataset(
            datafile,
            tokenizer=tokenizer,
            use_train_split=True,
            eval_start_p=eval_start_p,
            train_end_p=train_end_p,
            calibration_p=calibration_p,
            n_shot=ntrain,
            ensure_last_token_is_target=(not train_on_all_tokens),
            exclude_subjects=exclude_from_train,
            include_model_response=include_model_response,
            use_model_name_in_prompt=use_model_name_in_prompt,
            group_by_question=group_by_question,
            specialty_prompttype=specialty_prompttype,
            override_splits=override_splits
        )
        val_dataset = GradingDataset(
            datafile,
            tokenizer=tokenizer,
            use_train_split=False,
            eval_start_p=eval_start_p,
            train_end_p=train_end_p,
            calibration_p=-1,
            n_shot=ntrain,
            ensure_last_token_is_target=(not train_on_all_tokens),
            include_model_response=include_model_response,
            use_model_name_in_prompt=use_model_name_in_prompt,
            group_by_question=group_by_question,
            specialty_prompttype=specialty_prompttype,
            override_splits=override_splits
        )
    elif dataset_name == "CORRECTNESS_TUNE_RETRIEVAL":
        train_dataset_for_chroma_retrieval = GradingDataset(
            datafile,
            tokenizer=tokenizer,
            use_train_split=True,
            eval_start_p=eval_start_p,
            train_end_p=train_end_p,
            calibration_p=calibration_p,
            n_shot=ntrain,
            ensure_last_token_is_target=(not train_on_all_tokens),
            exclude_subjects=exclude_from_train,
            include_model_response=include_model_response,
            apply_chat_template=False,  # Important: no chat template for chroma source
            verbose=False,
            use_model_name_in_prompt=use_model_name_in_prompt,
            group_by_question=group_by_question,
            specialty_prompttype=specialty_prompttype,
            override_splits=override_splits
        )

        # Build training dataset with retrieval
        train_dataset = GradingDatasetWithRetrieval(
            datafile,
            tokenizer=tokenizer,
            use_train_split=True,
            eval_start_p=eval_start_p,
            train_end_p=train_end_p,
            calibration_p=calibration_p,
            n_shot=ntrain,
            ensure_last_token_is_target=(not train_on_all_tokens),
            exclude_subjects=exclude_from_train,
            include_model_response=include_model_response,
            chroma_source_dataset=train_dataset_for_chroma_retrieval,
            chroma_path=chroma_path,
            collection_name=collection_name,
            k=k_retrieve,
            has_no_overlap=False,
            verbalized=verbalized,
            use_reasoning=use_reasoning,
            use_model_name_in_prompt=use_model_name_in_prompt,
            embedding_function=embedding_function,
            group_by_question=group_by_question,
            specialty_prompttype=specialty_prompttype,
            override_splits=override_splits
        )
        train_dataset.augment_with_retrieved_examples_batched()
        print(f"{len(train_dataset)=}")
        chroma_collection = train_dataset.get_chroma_collection()
        existing_embedding_function_instance = train_dataset.get_existing_embedding_function_instance() if embedding_function else None
        val_dataset = GradingDatasetWithRetrieval(
            datafile,
            tokenizer=tokenizer,
            use_train_split=False,
            eval_start_p=eval_start_p,
            train_end_p=train_end_p,
            calibration_p=-1,
            n_shot=ntrain,
            ensure_last_token_is_target=(not train_on_all_tokens),
            include_model_response=include_model_response,
            chroma_source_dataset=train_dataset_for_chroma_retrieval,
            collection_name=collection_name,
            k=k_retrieve,
            existing_chroma_collection=chroma_collection,
            verbalized=verbalized,
            use_reasoning=use_reasoning,
            use_model_name_in_prompt=use_model_name_in_prompt,
            embedding_function=embedding_function,
            existing_embedding_function_instance=existing_embedding_function_instance,
            group_by_question=group_by_question,
            specialty_prompttype=specialty_prompttype,
            override_splits=override_splits
        )
        val_dataset.augment_with_retrieved_examples_batched()
        print(f"{len(val_dataset)=}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if shuffle:
        train_dataset.shuffle()
        val_dataset.shuffle()
    print(f"Size of datasets {len(train_dataset)=} {len(val_dataset)=} {train_dataset} {val_dataset}")

    # Since the data has been constructed, we no longer need the chroma_source_dataset object.
    if hasattr(train_dataset, 'chroma_source_dataset'):
        train_dataset.chroma_source_dataset = None
    if hasattr(val_dataset, 'chroma_source_dataset'):
        val_dataset.chroma_source_dataset = None
    # Also delete the existing_embedding_function_instance if we used one
    try: 
        train_dataset.existing_embedding_function_instance.unload()
        val_dataset.existing_embedding_function_instance.unload()
    except Exception as e:
        print(f"No custom Embedding function to unload: {e}")
    
    # Save to path if specified
    if dataset_save_path:
        raise Exception("Not Implemented")
    
    return train_dataset, val_dataset


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory.
    Returns the path to the latest checkpoint or None if no checkpoints found.
    """
    import glob
    import os

    if not os.path.exists(checkpoint_dir):
        return None

    # Look for checkpoint directories
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        return None

    # Sort by checkpoint number and return the latest
    def extract_checkpoint_number(path):
        try:
            return int(path.split("checkpoint-")[-1])
        except (ValueError, IndexError):
            return -1

    latest_checkpoint = max(checkpoints, key=extract_checkpoint_number)
    return latest_checkpoint
