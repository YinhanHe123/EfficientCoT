from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import utils
import torch
from tqdm import tqdm
import os
import deepspeed

def run_inference_with_deepspeed(contemp_generator, dataset, teacher_model_name, config, local_rank):
    """
    Run inference with the trained contemplation generator using DeepSpeed for the teacher model

    Args:
        contemp_generator: Trained contemplation generator
        dataset: Evaluation dataset
        teacher_model_name: Name of the teacher model
        config: Experiment configuration
        local_rank: Local rank for distributed inference
    """
    # Initialize DeepSpeed distributed environment if not already initialized
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed()

    # Set up device
    device = f"cuda:{local_rank}"
    contemp_generator = contemp_generator.to(device)
    contemp_generator.eval()

    # Load teacher LLM for generating answers
    # We'll use DeepSpeed inference for the teacher model
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # DeepSpeed Inference configuration for the teacher model
    ds_inference_config = {
        "tensor_parallel": {
            "tp_size": torch.cuda.device_count()  # Use all available GPUs
        },
        "dtype": "fp16",  # Use half precision for efficiency
        "replace_with_kernel_inject": True,
        "injection_policy": {
            "attention": "auto",
            "mlp": "auto"
        }
    }

    # Load the teacher model with DeepSpeed inference
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

    # Initialize DeepSpeed inference engine for the teacher model
    teacher_model = deepspeed.init_inference(
        teacher_model,
        mp_size=torch.cuda.device_count(),
        dtype=torch.float16,
        injection_policy={
            'attention': 'auto',
            'mlp': 'auto'
        },
        replace_with_kernel_inject=True
    )

    # Create output directory for results
    result_dir = f"{config.result_path}/{config.experiment_name}"
    utils.create_directory(result_dir)

    # For parallel inference, we'll split the dataset among ranks
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Split dataset across ranks
    samples_per_rank = len(dataset) // world_size
    start_idx = rank * samples_per_rank
    end_idx = (rank + 1) * samples_per_rank if rank < world_size - 1 else len(dataset)
    rank_dataset = [dataset[i] for i in range(start_idx, end_idx)]

    results = []

    with torch.no_grad():
        for sample in tqdm(rank_dataset, desc=f"Running inference (rank {rank})"):
            query = sample["query"]

            # Generate contemplation tokens hidden states (now acting as input embeddings)
            query_inputs = contemp_generator.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_seq_length // 2  # Leave room for the answer
            ).to(device)

            contemp_states = contemp_generator(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )

            # Prepare prompt with query
            prompt = f"Question: {query}\nAnswer:"

            # for debugging
            prompt = [
                {'role': "user", "content": prompt}
            ]
            teacher_tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{ '\n\nHuman: ' + message['content'] +  eos_token }}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\nAssistant: ' }}{% endif %}"
            prompt = teacher_tokenizer.apply_chat_template(prompt, tokenize=False)

            # Tokenize the prompt
            prompt_tokens = teacher_tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=config.max_seq_length // 2  # Leave room for the answer
            ).to(device)

            # Define a special token to separate prompt from contemplation
            separator_id = teacher_tokenizer.eos_token_id

            # Create input by concatenating prompt tokens with a separator
            input_ids = torch.cat([
                prompt_tokens.input_ids,
                torch.tensor([[separator_id]], device=device)
            ], dim=1)

            # Calculate the lengths for proper positioning
            prompt_length = input_ids.size(1)
            contemp_len = min(contemp_states.size(1), config.max_contemp_tokens)
            total_seq_length = prompt_length + contemp_len

            # For DeepSpeed inference, we need a different approach to inject the contemplation states
            # We'll first get the embeddings for all tokens
            inputs_embeds = teacher_model.get_input_embeddings()(input_ids)

            # Concatenate with contemplation states
            combined_embeds = torch.cat([
                inputs_embeds,
                contemp_states[:, -contemp_len:, :]
            ], dim=1)

            # Create proper attention mask
            attention_mask = torch.ones(
                (1, total_seq_length),
                dtype=torch.long,
                device=device
            )

            # Create position IDs
            position_ids = torch.arange(
                total_seq_length,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            # Generate output using inputs_embeds instead of input_ids
            outputs = teacher_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=120 + total_seq_length,  # Account for the input length
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            # Decode only the generated part
            answer = teacher_tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)

            if rank == 0:  # Only print on main process to avoid clutter
                print(f"Query: {query}\nAnswer: {answer}\n")

            result = {
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            }
            results.append(result)

    # Gather results from all ranks
    all_results = []
    for r in range(world_size):
        if r == rank:
            # Convert our results to tensor
            results_size = torch.tensor([len(results)], device=device)
        else:
            # Placeholder for other ranks
            results_size = torch.tensor([0], device=device)

        # Broadcast result size
        torch.distributed.broadcast(results_size, r)

        # If this rank has results to send
        if r == rank:
            # Send each result to all other ranks
            for result in results:
                # Convert to JSON string first
                result_str = utils.json.dumps(result)
                # Convert to tensor of bytes
                result_tensor = torch.tensor(bytearray(result_str.encode('utf-8')), dtype=torch.uint8, device=device)
                # Send tensor size first
                size_tensor = torch.tensor([len(result_tensor)], device=device)
                torch.distributed.broadcast(size_tensor, r)
                # Create padding if needed to ensure fixed size broadcast
                padded_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=device)
                padded_tensor[:len(result_tensor)] = result_tensor
                torch.distributed.broadcast(padded_tensor, r)
                # Collect on rank 0
                if rank == 0:
                    all_results.append(result)
        else:
            # Receive results from the current sender rank
            for _ in range(results_size.item()):
                # Receive size first
                size_tensor = torch.tensor([0], device=device)
                torch.distributed.broadcast(size_tensor, r)
                # Create tensor to receive data
                padded_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=device)
                torch.distributed.broadcast(padded_tensor, r)
                # Convert back to result dict
                if rank == 0:  # Only rank 0 needs to process the results
                    result_str = bytes(padded_tensor.tolist()).decode('utf-8')
                    result = utils.json.loads(result_str)
                    all_results.append(result)

    # Save results to file (only on rank 0)
    if rank == 0:
        # if path not exist, create it
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Save results to file
        utils.save_json(all_results, f"{result_dir}/inference_results.json")

    # Make sure all processes have completed before returning
    torch.distributed.barrier()

    return results