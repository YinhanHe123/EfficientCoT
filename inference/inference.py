def run_inference(contemp_generator, dataset, teacher_model_name, config):
    device = utils.get_device()
    contemp_generator = contemp_generator.to(device)
    contemp_generator.eval()

    # Load teacher LLM for generating answers
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Create output directory for results
    result_dir = f"{config.result_path}/{config.experiment_name}"
    utils.create_directory(result_dir)

    results = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Running inference"):
            query = sample["query"]

            # Generate contemplation tokens hidden states
            query_inputs = contemp_generator.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            contemp_states = contemp_generator(
                query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask
            )

            # Prepare prompt with query
            prompt = f"Question: {query}\nAnswer:"

            # Tokenize prompt
            prompt_inputs = teacher_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            # Generate answer using teacher model
            outputs = teacher_model.generate(
                prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            answer = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.replace(prompt, "").strip()

            result = {
                "query": query,
                "ground_truth": sample.get("answer", ""),
                "prediction": answer
            }

            results.append(result)

    # Save results to file
    utils.save_json(results, f"{result_dir}/inference_results.json")

    return results