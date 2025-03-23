#!/usr/bin/env python3
import os
import json
import argparse
import time
from typing import List, Dict, Any
import requests
from tqdm import tqdm
from datasets import load_dataset

class ReasoningPairsGenerator:
    """
    Generate detailed and condensed reasoning pairs for raw problems using ChatGPT-4o-mini.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the generator with API credentials.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def generate_reasoning(self, query: str) -> str:
        """
        Generate detailed reasoning for a given query.

        Args:
            query: The mathematical problem to solve

        Returns:
            Detailed step-by-step reasoning
        """
        messages = [
            {"role": "system", "content": "You are a math tutor who provides detailed step-by-step solutions to math problems."},
            {"role": "user", "content": f"Problem: {query}\n\nPlease solve this step-by-step, showing all your work."}
        ]

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                }
            )
            response.raise_for_status()
            reasoning = response.json()["choices"][0]["message"]["content"]
            return reasoning
        except Exception as e:
            print(f"Error generating detailed reasoning: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"API response: {response.text}")
            return ""

    def generate_condensed_reasoning(self, query: str, original_reasoning: str) -> str:
        """
        Generate condensed version of the original reasoning.

        Args:
            query: The original problem
            original_reasoning: The detailed reasoning to condense

        Returns:
            Condensed reasoning
        """
        messages = [
            {"role": "system", "content": "You are an AI that creates concise summaries of mathematical reasoning."},
            {"role": "user", "content": f"Problem: {query}\n\nDetailed Solution: {original_reasoning}\n\nPlease condense the above solution into a brief but complete chain of reasoning. Be concise but maintain accuracy."}
        ]

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                }
            )
            response.raise_for_status()
            condensed = response.json()["choices"][0]["message"]["content"]
            return condensed
        except Exception as e:
            print(f"Error generating condensed reasoning: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"API response: {response.text}")
            return ""

    def generate_answer(self, query: str, reasoning: str) -> str:
        """
        Extract or generate the final answer based on the reasoning.

        Args:
            query: The original problem
            reasoning: The reasoning from which to extract the answer

        Returns:
            The final answer
        """
        messages = [
            {"role": "system", "content": "You are an AI that extracts the final numerical answer from mathematical reasoning."},
            {"role": "user", "content": f"Problem: {query}\n\nReasoning: {reasoning}\n\nPlease extract or calculate the final numerical answer only. Return only the number or mathematical expression that represents the final answer."}
        ]

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                }
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            # Remove any extra text, keeping only the answer
            answer = answer.strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"API response: {response.text}")
            return ""

    def create_reasoning_pair(self, query: str) -> Dict[str, Any]:
        """
        Generate a pair of original and condensed reasoning for a query.

        Args:
            query: The mathematical problem to solve

        Returns:
            Dictionary with query, original reasoning, condensed reasoning, and answer
        """
        original_reasoning = self.generate_reasoning(query)
        time.sleep(1)  # Rate limiting

        condensed_reasoning = self.generate_condensed_reasoning(query, original_reasoning)
        time.sleep(1)  # Rate limiting

        answer = self.generate_answer(query, original_reasoning)

        return {
            "query": query,
            "original_reasoning": original_reasoning,
            "condensed_reasoning": condensed_reasoning,
            "answer": answer
        }

    def create_dataset(self, queries: List[str], output_file: str = "reasoning_pairs.json"):
        """
        Create a dataset of reasoning pairs from a list of queries and save to JSON.
        Appends to existing file if it exists to avoid losing previous work.

        Args:
            queries: List of queries to generate reasoning for
            output_file: Path to output JSON file
        """
        # Load existing dataset if the file exists
        dataset = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                print(f"Loaded existing dataset with {len(dataset)} examples")

                # Create a set of existing queries to avoid duplicates
                existing_queries = {item['query'] for item in dataset}

                # Filter out queries that have already been processed
                new_queries = [q for q in queries if q not in existing_queries]
                print(f"Found {len(queries) - len(new_queries)} already processed queries")
                queries = new_queries

                print(f"Will process {len(queries)} new queries")
            except Exception as e:
                print(f"Error loading existing dataset: {e}")
                print("Starting with an empty dataset")

        for i, query in enumerate(tqdm(queries, desc="Generating reasoning pairs")):
            try:
                pair = self.create_reasoning_pair(query)
                dataset.append(pair)

                # Save progress after every 5 examples or at the last one
                if (i + 1) % 5 == 0 or i == len(queries) - 1:
                    # Write to a temporary file first, then rename to avoid corruption
                    temp_file = f"{output_file}.temp"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, indent=2, ensure_ascii=False)

                    # Safely replace the original file
                    os.replace(temp_file, output_file)

                    print(f"Saved progress: {len(dataset)} total examples ({i+1}/{len(queries)} new examples)")

                # Rate limiting to avoid hitting API limits
                time.sleep(2)

            except Exception as e:
                print(f"Error processing query {i}: {e}")
                # Save progress even on failure
                temp_file = f"{output_file}.temp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, output_file)

        return dataset


def load_raw_dataset(limit: int = None):
    """
    Load raw dataset directly from Hugging Face datasets.

    Args:
        limit: Maximum number of examples to load

    Returns:
        List of questions from the raw dataset
    """
    # Load raw dataset from Hugging Face
    dataset = load_dataset("raw", "main")

    # Extract questions from the training set
    train_data = dataset["train"]

    # Format the queries
    queries = []
    for i, item in enumerate(train_data):
        queries.append(item["question"])

    # Return all queries (the slicing will be handled in main())
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning pairs for raw dataset using ChatGPT")
    parser.add_argument("--output_file", type=str, default="reasoning_pairs.json", help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=500, help="Limit number of examples to process")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index in the raw dataset")
    parser.add_argument("--retry_failed", action="store_true", help="Try to reprocess examples that failed before")

    args = parser.parse_args()

    # Hardcoded API key directly in the code
    api_key = "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"  # Replace with your actual OpenAI API key

    # Load raw dataset directly from Hugging Face
    try:
        # Load all queries within the range [start_index, start_index + limit)
        all_queries = load_raw_dataset()
        if args.limit:
            end_index = min(args.start_index + args.limit, len(all_queries))
            queries = all_queries[args.start_index:end_index]
        else:
            queries = all_queries[args.start_index:]

        print(f"Loaded {len(queries)} problems from raw dataset (starting at index {args.start_index})")
    except Exception as e:
        print(f"Error loading raw dataset: {e}")
        return

    # Check for existing output file and handle failed examples
    if args.retry_failed and os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # Find failed examples (those with empty fields)
            failed_examples = []
            for item in existing_data:
                if (not item.get('original_reasoning') or
                    not item.get('condensed_reasoning') or
                    not item.get('answer')):
                    failed_examples.append(item['query'])

            if failed_examples:
                print(f"Found {len(failed_examples)} failed examples to retry")
                queries = failed_examples
        except Exception as e:
            print(f"Error checking for failed examples: {e}")

    # Generate reasoning pairs
    generator = ReasoningPairsGenerator(api_key, model=args.model)
    generator.create_dataset(queries, args.output_file)

    print(f"Completed generation of reasoning pairs")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()