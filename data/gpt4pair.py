#!/usr/bin/env python3
import os
import json
import argparse
import time
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

class ReasoningPairsGenerator:
    """
    Generate detailed and condensed reasoning pairs for raw problems using ChatGPT-4o-mini.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = "key"):
        """
        Initialize the generator with API credentials.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
        """
        self.client = OpenAI(
            # base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            )
        self.model = model_name

    def generate_reasoning(self, query: str) -> str:
        """
        Generate detailed reasoning for a given query.

        Args:
            query: The mathematical problem to solve

        Returns:
            Detailed step-by-step reasoning
        """
         # Determine the type of problem
        is_commonsense_qa = "choices:" in query.lower()
        is_coin_flip = "coin" in query.lower() and "flip" in query.lower()
        is_strategyqa = query.lower().endswith(("true", "false", "yes", "no")) or any(indicator in query.lower() for indicator in ["would", "could", "should", "is it", "will", "does", "can"])
        is_logiqa = "context:" in query.lower() and "options:" in query.lower() and any(f"{i}." in query for i in range(4))
        is_multihopqa = len(query.split()) > 20 and any(indicator in query.lower() for indicator in ["what", "which", "where", "when", "who", "how"]) and not ("choices:" in query.lower() or "options:" in query.lower())

        if is_commonsense_qa:
            messages = [
                {"role": "system", "content": "You are an AI that provides step-by-step reasoning for multiple-choice commonsense questions."},
                {"role": "user", "content": f"Problem: {query}\n\nPlease think through this question step by step. Consider what each answer option means and why it would or wouldn't be appropriate. At the end, provide your answer as a single letter (A, B, C, D, or E)."}
            ]
        elif is_coin_flip:
            messages = [
                {"role": "system", "content": "You are an AI that explains coin flip logical problems."},
                {"role": "user", "content": f"Problem: {query}\n\nPlease solve this problem by tracking the state of the coin at each step. You only need to consider whether the coin is heads up or tails up after each flip or non-flip. At the end, answer with 'yes' or 'no' to whether the coin is still heads up."}
            ]
        elif is_strategyqa:
            messages = [
                {"role": "system", "content": "You are an AI that excels at strategic reasoning and answering yes/no questions that require multi-step logical thinking."},
                {"role": "user", "content": f"Question: {query}\n\nPlease reason through this step by step. Consider all relevant facts, implications, and logical connections. Break down the problem into smaller parts if needed, and clearly explain your reasoning process. At the end, provide your answer as either 'true' or 'false'."}
            ]
        elif is_logiqa:
            messages = [
                {"role": "system", "content": "You are an AI expert in logical reasoning and critical thinking. You excel at analyzing complex logical relationships and drawing valid conclusions."},
                {"role": "user", "content": f"Logical Reasoning Problem: {query}\n\nPlease analyze this logical reasoning problem step by step:\n1. First, identify the key premises and logical structure in the context\n2. Understand what the question is asking\n3. Evaluate each option systematically using logical principles\n4. Eliminate invalid options and identify the most logically sound answer\n5. Provide your final answer as the option number (0, 1, 2, or 3)."}
            ]
        elif is_multihopqa:
            messages = [
                {"role": "system", "content": "You are an AI that specializes in multi-hop reasoning and complex question answering. You excel at connecting multiple pieces of information to reach accurate conclusions."},
                {"role": "user", "content": f"Multi-hop Question: {query}\n\nThis question requires multi-step reasoning. Please approach it systematically:\n1. Break down the question into sub-questions or information requirements\n2. Identify what information is needed at each step\n3. Connect the pieces of information logically\n4. Work through each hop in the reasoning chain\n5. Synthesize your findings to provide a comprehensive and accurate final answer."}
            ]
        else:
            # Default math tutor prompt
            messages = [
                {"role": "system", "content": "You are a math tutor who provides detailed step-by-step solutions to math problems."},
                {"role": "user", "content": f"Problem: {query}\n\nPlease solve this step-by-step, showing all your work."}
            ]


        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            reasoning = completion.choices[0].message.content
            return reasoning
        except Exception as e:
            print(f"Error generating detailed reasoning: {e}")
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
            {"role": "user", "content": f"Problem: {query}\n\nDetailed Solution: {original_reasoning}\n\nPlease condense the above solution into a brief but complete chain of reasoning. Be concise but maintain accuracy within 10 words."}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            condensed = completion.choices[0].message.content
            return condensed
        except Exception as e:
            print(f"Error generating condensed reasoning: {e}")
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
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            answer = completion.choices[0].message.content
            answer = answer.strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""

    def create_reasoning_pair(self, query: str, given_reasoning:str, given_answer:str) -> Dict[str, Any]:
        """
        Generate a pair of original and condensed reasoning for a query.

        Args:
            query: The mathematical problem to solve

        Returns:
            Dictionary with query, original reasoning, condensed reasoning, and answer
        """
        if given_reasoning:
            original_reasoning = given_reasoning
        else:
            original_reasoning = self.generate_reasoning(query)
            time.sleep(0.5)  # Rate limiting

        condensed_reasoning = self.generate_condensed_reasoning(query, original_reasoning)
        time.sleep(0.5)  # Rate limiting

        if given_answer:
            answer = given_answer
        else:
            answer = self.generate_answer(query, original_reasoning)

        return {
            "query": query,
            "original_reasoning": original_reasoning,
            "condensed_reasoning": condensed_reasoning,
            "answer": answer
        }

    def create_dataset(self, queries: List[str], reasonings: List[str], answers: List[str], output_file: str = "reasoning_pairs.json"):
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

        for i in tqdm(range(len(queries)), desc="Generating reasoning pairs"):
            try:
                pair = self.create_reasoning_pair(queries[i], reasonings[i] if reasonings else None, answers[i] if answers else None)
                dataset.append(pair)

                # Save progress after every 5 examples or at the last one
                if (i + 1) % 5 == 0 or i == len(queries) - 1:
                    # Write to a temporary file first, then rename to avoid corruption
                    temp_file = f"{output_file}.temp"
                    if not os.path.exists(temp_file):
                        # enforce creating the new file and nonexist folders
                        os.makedirs(os.path.dirname(temp_file), exist_ok=True)

                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, indent=2, ensure_ascii=False)

                    # Safely replace the original file
                    os.replace(temp_file, output_file)

                    print(f"Saved progress: {len(dataset)} total examples ({i+1}/{len(queries)} new examples)")

                # Rate limiting to avoid hitting API limits
                time.sleep(1)

            except Exception as e:
                print(f"Error processing query {i}: {e}")
                # Save progress even on failure
                temp_file = f"{output_file}.temp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, output_file)

        return dataset


