import torch
from tqdm import tqdm
import os
import time
# import CCoT related models
from models.ccot_model import CCoTModel, CCOTDecodeModel
import utils.utils as utils
import pdb
from datasets import Dataset as HFDataset
import gc
import sys

def prepare_ccot_decode_dataset(queries, answers, ccot_model, tokenizer, device, max_length=120, cotrain_mode=False):
    """
    Prepare a dataset for training the decode model with HuggingFace Dataset
    """
    # Prepare data dictionaries
    input_ids_list = []
    attention_mask_list = []
    contemp_states_list = []
    labels_list = []

    # Process each query-answer pair
    for query, answer in tqdm(zip(queries, answers), total=len(queries), desc="Preparing decode dataset"):
        # Tokenize the query
        query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length",
                                max_length=max_length).to(device)

        # Generate contemplation tokens
        if cotrain_mode:
            contemp_states = ccot_model(query_inputs.input_ids,
                                      attention_mask=query_inputs.attention_mask,
                                      max_contemplation_tokens=10)
        else:
            with torch.no_grad():
                contemp_states = ccot_model(query_inputs.input_ids,
                                        attention_mask=query_inputs.attention_mask)

        # Extract reasoning and final answer
        reasoning_parts = answer.split('####')
        reasoning = reasoning_parts[0].strip()

        # The final answer comes after ####
        final_answer = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else ""


        # Tokenize the answer
        answer_inputs = tokenizer(final_answer, return_tensors="pt", truncation=True, padding="max_length",
                                 max_length=5)

        # # Combine query and answer for label creation
        # combined = f"{query} {answer}"
        # combined_inputs = tokenizer(combined, return_tensors="pt", truncation=True, padding="max_length",
        #                            max_length=max_length).to(device)

        # # Create labels (shift by 1)
        # labels = combined_inputs.input_ids.clone()
        # labels[:, :query_inputs.input_ids.size(1)] = -100  # Mask out the query part

        # Store the data
        if cotrain_mode:
            input_ids_list.append(query_inputs.input_ids)
            attention_mask_list.append(query_inputs.attention_mask)
            contemp_states_list.append(contemp_states)
            labels_list.append(answer_inputs.input_ids.squeeze())
        else:
            input_ids_list.append(query_inputs.input_ids.cpu().numpy())
            attention_mask_list.append(query_inputs.attention_mask.cpu().numpy())
            contemp_states_list.append(contemp_states.cpu().numpy())
            labels_list.append(answer_inputs.input_ids.squeeze().numpy())

    # Create HuggingFace Dataset


    if cotrain_mode:
        dataset_dict = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "contemp_states": torch.stack(contemp_states_list),
            "labels": torch.stack(labels_list)}
        return dataset_dict
    else:
        dataset_dict = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "contemp_states": contemp_states_list,
        "labels": labels_list
    }
        return HFDataset.from_dict(dataset_dict)

def get_decode_dataset(train_dataset, eval_dataset, ccot_model, tokenizer, device, cotrain_mode=False):
    # Extract training data
    queries = [item["question"] for item in train_dataset.dataset]
    answers = [item["answer"] for item in train_dataset.dataset]

    # Extract evaluation data
    eval_queries = [item["question"] for item in eval_dataset.dataset]
    eval_answers = [item["answer"] for item in eval_dataset.dataset]

    # Create datasets
    # Create datasets using the new function
    train_decode_dataset = prepare_ccot_decode_dataset(
        queries=queries,
        answers=answers,
        ccot_model=ccot_model,
        tokenizer=tokenizer,
        device=device,
        cotrain_mode=cotrain_mode
    )

    eval_decode_dataset = prepare_ccot_decode_dataset(
        queries=eval_queries,
        answers=eval_answers,
        ccot_model=ccot_model,
        tokenizer=tokenizer,
        device=device,
        cotrain_mode=cotrain_mode
    )

    return train_decode_dataset, eval_decode_dataset


def run_ccot_baseline(train_dataset, eval_dataset, model_config, experiment_config):
    """
    Compressed Chain of Thought baseline
    Based on: Cheng & Van Durme "Compressed chain of thought: Efficient reasoning through dense representations"

    Args:
        dataset: Dataset for evaluation
        model_config: Model configuration
        experiment_config: Experiment configuration

    Returns:
        List of prediction results
    """
    device = experiment_config.device

    # Check for trained models
    ccot_model_path = os.path.join(experiment_config.model_save_path, "ccot_model")
    decode_model_path = os.path.join(experiment_config.model_save_path, "ccot_decode_model")
    decode_dataset_path = os.path.join(experiment_config.model_save_path, "ccot_decode_dataset")
    # if ccot model not exist, do it, if done, load it
    # elif it's already there, load it.
    if experiment_config.ccot_stage == "encode":
        if not os.path.exists(ccot_model_path+'/config.pt'):
            # Train models if not available
            print("No pre-trained CCOT models found. Training models...")

            # First, train the CCOT model
            from training.train_ccot import train_ccot_model, train_ccot_decode_model


            os.makedirs(ccot_model_path, exist_ok=True)
            ccot_model = train_ccot_model(
                base_model_name=model_config.teacher_model_name,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_path=ccot_model_path,
                compression_ratio=experiment_config.compression_ratio,
                autoregressive_layer=experiment_config.autoregressive_layer,
                learning_rate=experiment_config.learning_rate,
                num_epochs_per_layer=1,
                batch_size=experiment_config.batch_size,
                device=device
            )
            ccot_model.eval()
        elif not os.path.exists(decode_dataset_path+"/train"):
            # Load pre-trained models
            print("CCOT model already trained!")
        sys.exit()

    if experiment_config.ccot_stage == "prepare_decode_data":
        if not os.path.exists(decode_dataset_path+"/train"):
            # Create decode dataset
            try:
                ccot_model = CCoTModel.from_pretrained(ccot_model_path)
                ccot_model = ccot_model.to(device)
                ccot_model.eval()
            except:
                print("CCOT model not found. Please train the CCOT model first (i.e., ccot_stage=encode).")
            print("Preparing decode dataset...")
            train_decode_dataset, eval_decode_dataset = get_decode_dataset(train_dataset, eval_dataset, ccot_model, ccot_model.tokenizer, device)
            # Save the datasets
            os.makedirs(decode_dataset_path+"/train", exist_ok=True)
            os.makedirs(decode_dataset_path+"/eval", exist_ok=True)
            train_decode_dataset.save_to_disk(decode_dataset_path + "/train")
            eval_decode_dataset.save_to_disk(decode_dataset_path + "/eval")
        else:
            # Load decode dataset
            print("Decode dataset already prepared!")
        sys.exit()

    if experiment_config.ccot_stage == "decode":
        if not os.path.exists(decode_model_path+'/config.pt'):
            # Train the decoder model
            print("Training decode model...")
            # load the decode dataset
            try:
                train_decode_dataset = HFDataset.load_from_disk(decode_dataset_path + "/train")
                eval_decode_dataset = HFDataset.load_from_disk(decode_dataset_path + "/eval")
            except:
                print("Decode dataset not found. Please prepare the decode dataset first (i.e., ccot_stage=prepare_decode_data).")
            from training.train_ccot import train_ccot_decode_model
            os.makedirs(decode_model_path, exist_ok=True)
            decode_model = train_ccot_decode_model(
                base_model_name=model_config.teacher_model_name,
                train_decode_dataset=train_decode_dataset,
                eval_decode_dataset=eval_decode_dataset,
                output_path=decode_model_path,
                learning_rate=experiment_config.learning_rate / 10,  # Lower learning rate for decoder
                num_epochs=3,
                batch_size=2,
                device=device
            )
            decode_model.eval()
        else:
            print("Decode model already trained!")
        sys.exit()

    if experiment_config.ccot_stage == "evaluate":
        print("Predicting on evaluation dataset...")
        # Load pre-trained models
        try:
            results = []
            contemplation_states_list = []
            contemp_gen_time = []
            decode_time = []
            # Check if cotuned models are available
            cotrain_output_path = os.path.join(experiment_config.model_save_path, "cotrained")
            cotrained_ccot_path = os.path.join(cotrain_output_path, "ccot_model")
            cotrained_decode_path = os.path.join(cotrain_output_path, "ccot_decode_model")
            # If cotuned models exist, use them instead of the original ones
            ccot_model_path = cotrained_ccot_path if os.path.exists(cotrained_ccot_path+'/config.pt') else ccot_model_path
            decode_model_path = cotrained_decode_path if os.path.exists(cotrained_decode_path+'/config.pt') else decode_model_path
            with torch.no_grad():
                ccot_model = CCoTModel.from_pretrained(ccot_model_path)
                ccot_model = ccot_model.to(device)
                ccot_model.eval()

                for sample in tqdm(eval_dataset, desc="Getting contemplations"):
                    query = sample["query"]

                    # Tokenize input
                    input_text = f"Question: {query}\nAnswer:"
                    inputs = ccot_model.tokenizer(
                        input_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=experiment_config.max_seq_length # Leave room for answer
                    ).to(device)

                    # Generate contemplation tokens
                    # Prepare input
                    start_time = time.time()
                    contemplation_states = ccot_model(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_contemplation_tokens=experiment_config.max_contemp_tokens # This line should be removed after THE ARG IS REMOVED
                    )
                    end_time = time.time()
                    print(f"Contemplation generation time: {end_time - start_time:.2f} seconds")
                    contemplation_states_list.append(contemplation_states.cpu().numpy())

                    contemp_generation_time = end_time - start_time
                    contemp_gen_time.append(contemp_generation_time)

                del ccot_model
                gc.collect()
                torch.cuda.empty_cache()
                print('\n\n\n')
                # Load pre-trained decode model
                decode_model = CCOTDecodeModel.from_pretrained(decode_model_path)
                decode_model = decode_model.to(device)
                decode_model.eval()
                for sample in tqdm(eval_dataset, desc="decoding"):
                    query = sample["query"]

                    # Tokenize input
                    input_text = f"Question: {query}\n Generate the answer directly. Answer:"
                    inputs = decode_model.tokenizer(
                        input_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=experiment_config.max_seq_length
                    ).to(device)

                    # concate with corresponding contemplation states
                    contemplation_states = torch.tensor(contemplation_states_list.pop(0)).to(device)
                    # contemplation_states = torch.rand(contemplation_states_list.pop(0).shape).to(device)
                    # Generate contemplation tokens
                     # Prepare input
                    # start_time = time.time()

                    original_prepare_inputs = decode_model.model.prepare_inputs_for_generation

                    # # Remove the first_call mechanism and modify the prepare_inputs function
                    def modified_prepare_inputs(input_ids, past_key_values=None, **kwargs):
                        # If this is the first call (no past_key_values)
                        if len(past_key_values.key_cache) == 0:
                            # Get the embeddings from the model's embedding layer
                            inputs_embeds = decode_model.model.get_input_embeddings()(input_ids)

                            # Create a new inputs_embeds by concatenating with contemp_states
                            contemp_len = min(contemplation_states.shape[1], experiment_config.max_contemp_tokens)


                            total_seq_length = input_ids.shape[1] + contemp_len
                            combined_embeds = torch.cat([
                                inputs_embeds,
                                contemplation_states[:, -contemp_len:, :]
                            ], dim=1)



                            # Create a proper attention mask that covers both parts
                            attention_mask = torch.ones(
                                (1, total_seq_length),
                                dtype=torch.long,
                                device=device
                            )

                            # Create position ids that account for both parts
                            position_ids = torch.arange(
                                total_seq_length,
                                dtype=torch.long,
                                device=device
                            ).unsqueeze(0)

                            # Return the combined inputs with proper positioning
                            kwargs.pop('attention_mask', None)
                            kwargs.pop('position_ids', None)

                            return {
                                'inputs_embeds': combined_embeds,
                                'attention_mask': attention_mask,
                                'position_ids': position_ids,
                                **kwargs
                            }

                        # For subsequent calls, adjust the past_key_values to include the effect of contemp_states
                        else:
                            # The model will continue generating based on the KV cache that already includes
                            # the effect of the contemplation states from the first call
                            return original_prepare_inputs(input_ids, past_key_values=past_key_values, **kwargs)


                    # # Replace the prepare_inputs_for_generation method temporarily
                    decode_model.model.prepare_inputs_for_generation = modified_prepare_inputs
                    # # teacher_model.prepare_inputs_for_generation = original_prepare_inputs # for debugging
                    # # Generate answer with the modified approach
                    start_time = time.time()
                    outputs = decode_model.model.generate(
                        inputs.input_ids,
                        max_length=15 + inputs.input_ids.size(1) + experiment_config.max_contemp_tokens,  # Account for the input length
                        # max_length=15 + inputs.input_ids.size(1)+2,  # Account for the input length
                        # max_new_tokens=20,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=True
                    )
                    # print('INPUT: ', decode_model.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True),'\n')
                    # print('OUTPUT: ', decode_model.tokenizer.decode(outputs[0], skip_special_tokens=True))
                    end_time = time.time()
                    decode_model.model.prepare_inputs_for_generation = original_prepare_inputs # change it back to original for the next sample in the loop

                    decode_time.append(end_time - start_time)
                    # Decode the output
                    response = decode_model.tokenizer.decode(
                        outputs[0][inputs.input_ids.size(1):],
                        skip_special_tokens=True
                    )
                    answer = response.replace(input_text, "").strip()

                    # Save result
                    results.append({
                        "query": query,
                        "ground_truth": sample.get("answer", ""),
                        "prediction": answer,
                        "generation_time": end_time - start_time
                    })
                del decode_model
                gc.collect()
                torch.cuda.empty_cache()

                # calculate total time
                total_time = torch.tensor(decode_time) + torch.tensor(contemp_gen_time)
                avg_time = total_time.mean().item()
                print(f"Average generation time: {avg_time:.2f} seconds, Average contemp generation time: {torch.tensor(contemp_gen_time).mean().item():.2f} seconds, Average decode time: {torch.tensor(decode_time).mean().item():.2f} seconds")

                # Save results
                results_dir = os.path.join(experiment_config.result_path, "ccot")
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                # Calculate average generation time
                # avg_time = total_time / len(eval_dataset)

                # Add summary statistics
                summary = {
                    "avg_generation_time": avg_time,
                    "num_samples": len(eval_dataset),
                    "compression_ratio": experiment_config.compression_ratio,
                    "autoregressive_layer": experiment_config.autoregressive_layer
                }

                # Save results with summary
                utils.save_json({"results": results, "summary": summary}, f"{results_dir}/inference_results.json")

                print(f"CCOT baseline completed. Average generation time: {avg_time:.2f} seconds")

                return results
        except:
            print("CCOT / Decode data/ Decode model not found. Please train them first (e.g, ccot_stage=encode/prepare_decode_data/decode).")

    if experiment_config.ccot_stage == "cotrain_encode_decode":
        from training.train_ccot import cotrain_encode_decode

        ccot_model_path = os.path.join(experiment_config.model_save_path, "ccot_model")
        # Check if prerequisites exist
        if not os.path.exists(ccot_model_path+'/config.pt'):
            print("CCOT model not found. Please train the CCOT model first (i.e., ccot_stage=encode).")
            sys.exit()

        # Create output directory for the cotuned model
        cotrain_output_path = os.path.join(experiment_config.model_save_path, "cotrained")
        os.makedirs(cotrain_output_path, exist_ok=True)

        print("Cotraining encoder and decoder models...")
        ccot_model, decode_model = cotrain_encode_decode(
            ccot_model_path=ccot_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=cotrain_output_path,
            autoregressive_layer=experiment_config.autoregressive_layer,
            learning_rate=experiment_config.learning_rate / 20,  # Lower learning rate for cotuning
            num_epochs=3,
            batch_size=2,
            device=device
        )

        print("Cotraining completed! The cotuned models are saved at:")
        print(f"- Encoder: {os.path.join(cotrain_output_path, 'ccot_model')}")
        print(f"- Decoder: {os.path.join(cotrain_output_path, 'ccot_decode_model')}")
        print("Cotraining completed!")
        sys.exit()
