import torch
from tqdm import tqdm
import os
import time
# import CCoT related models
from models.ccot_model import CCoTModel, CCOTDecodeModel
import utils.utils as utils
import pdb 

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
    
    if os.path.exists(ccot_model_path) and os.path.exists(decode_model_path):
        # Load pre-trained models
        print("Loading pre-trained CCOT models...")
        ccot_model = CCoTModel.from_pretrained(ccot_model_path)
        ccot_model = ccot_model.to(device)
        ccot_model.eval()
        
        decode_model = CCOTDecodeModel.from_pretrained(decode_model_path)
        decode_model = decode_model.to(device)
        decode_model.eval()
    elif os.path.exists(ccot_model_path):
        # Load pre-trained models
        print("Loading pre-trained CCOT models...")
        from training.train_ccot import train_ccot_decode_model
        ccot_model = CCoTModel.from_pretrained(ccot_model_path).to(device)
        
        # Train the decoder model
        decode_model = train_ccot_decode_model(
            base_model_name=model_config.teacher_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ccot_model=ccot_model,
            output_path=decode_model_path,
            learning_rate=experiment_config.learning_rate / 10,  # Lower learning rate for decoder
            num_epochs=5,
            batch_size=experiment_config.batch_size // 2,  # Smaller batch size to avoid OOM
            device=device
        )
        decode_model.eval()
    else:
        # Train models if not available
        print("No pre-trained CCOT models found. Training models...")
        
        # First, train the CCOT model
        from training.train_ccot import train_ccot_model, train_ccot_decode_model
        
        # Train the CCOT model, commented for debugging decode model
        ccot_model = train_ccot_model(
            base_model_name=model_config.teacher_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_path=ccot_model_path,
            compression_ratio=experiment_config.compression_ratio,
            autoregressive_layer=experiment_config.autoregressive_layer,
            learning_rate=experiment_config.learning_rate,
            num_epochs_per_layer=3,
            batch_size=experiment_config.batch_size,
            device=device
        )
        
        # Train the decoder model
        decode_model = train_ccot_decode_model(
            base_model_name=model_config.teacher_model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ccot_model=ccot_model,
            output_path=decode_model_path,
            learning_rate=experiment_config.learning_rate / 10,  # Lower learning rate for decoder
            num_epochs=5,
            batch_size=experiment_config.batch_size // 2,  # Smaller batch size to avoid OOM
            device=device
        )
        
        ccot_model.eval()
        decode_model.eval()
    
    print("Predicting on evaluation dataset...")
    # Initialize tokenizer
    tokenizer = ccot_model.tokenizer
    
    # Run inference
    results = []
    total_time = 0
    
    with torch.no_grad():
        for sample in tqdm(eval_dataset, desc="Running CCOT baseline"):
            query = sample["question"]
            
            # Prepare input
            start_time = time.time()
            
            # Tokenize input
            input_text = f"Question: {query}\nAnswer:"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=experiment_config.max_seq_length // 2  # Leave room for answer
            ).to(device)
            
            # Generate contemplation tokens
            contemplation_states = ccot_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            # Generate answer with decode model
            outputs = decode_model.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                # Use the decode model to generate
                prefix_state=contemplation_states  # Provide contemplation states for conditioning
            )
            
            # Calculate generation time
            end_time = time.time()
            generation_time = end_time - start_time
            total_time += generation_time
            
            # Decode the output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(input_text, "").strip()
            
            # Save result
            results.append({
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer,
                "generation_time": generation_time
            })
    
    # Save results
    results_dir = os.path.join(experiment_config.result_path, "ccot")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Calculate average generation time
    avg_time = total_time / len(eval_dataset)
    
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