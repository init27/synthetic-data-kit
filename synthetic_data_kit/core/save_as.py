# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for saving file format

import os
import json
import glob
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List

from synthetic_data_kit.utils.format_converter import to_jsonl, to_alpaca, to_fine_tuning, to_chatml, to_hf_dataset
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format

def convert_format(
    input_path: str,
    output_path: str,
    format_type: str,
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
) -> str:
    """Convert data to different formats
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the output
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf" (Hugging Face dataset)
    
    Returns:
        Path to the output file or directory
    """
    # Load input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data based on known structures
    # Try to handle the case where we have QA pairs or conversations
    if "qa_pairs" in data:
        qa_pairs = data.get("qa_pairs", [])
    elif "filtered_pairs" in data:
        qa_pairs = data.get("filtered_pairs", [])
    elif "conversations" in data:
        conversations = data.get("conversations", [])
        qa_pairs = []
        for conv in conversations:
            if len(conv) >= 3 and conv[1]['role'] == 'user' and conv[2]['role'] == 'assistant':
                qa_pairs.append({
                    'question': conv[1]['content'],
                    'answer': conv[2]['content']
                })
    else:
        # If the file is just an array of objects, check if they look like QA pairs
        if isinstance(data, list):
            qa_pairs = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    qa_pairs.append(item)
        else:
            raise ValueError("Unrecognized data format - expected QA pairs or conversations")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # When using HF dataset storage format
    if storage_format == "hf":
        # For HF datasets, we need to prepare the data in the right structure
        if format_type == "jsonl":
            # For JSONL, just use the QA pairs directly
            formatted_pairs = qa_pairs
        elif format_type == "alpaca":
            # Format as Alpaca structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "instruction": pair["question"],
                    "input": "",
                    "output": pair["answer"]
                })
        elif format_type == "ft":
            # Format as OpenAI fine-tuning structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                })
        elif format_type == "chatml":
            # Format as ChatML structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                })
        else:
            raise ValueError(f"Unknown format type: {format_type}")
            
        # Save as HF dataset (Arrow format)
        return to_hf_dataset(formatted_pairs, output_path)
    
    # Standard JSON file storage format
    else:
        # Convert to the requested format using existing functions
        if format_type == "jsonl":
            return to_jsonl(qa_pairs, output_path)
        elif format_type == "alpaca":
            return to_alpaca(qa_pairs, output_path)
        elif format_type == "ft":
            return to_fine_tuning(qa_pairs, output_path)
        elif format_type == "chatml":
            return to_chatml(qa_pairs, output_path)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
            
def process_directory(
    input_dir: str,
    output_dir: str,
    format_type: str,
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> List[str]:
    """Process all files in a directory and convert to specified format
    
    Args:
        input_dir: Directory containing files to convert
        output_dir: Directory to save converted files
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf"
        parallel: Whether to process files in parallel (default) or sequentially
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of paths to output files
    """
    # Get all cleaned JSON files in the directory
    files = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and file_path.endswith('_cleaned.json'):
            files.append(file_path)
    
    if not files:
        raise ValueError(f"No cleaned files found in directory: {input_dir}")
    
    return process_multiple_files(
        input_files=files,
        output_dir=output_dir,
        format_type=format_type,
        config=config,
        storage_format=storage_format,
        parallel=parallel,
        max_workers=max_workers
    )

def process_multiple_files(
    input_files: List[str],
    output_dir: str,
    format_type: str,
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> List[str]:
    """Process multiple files and convert to specified format
    
    Args:
        input_files: List of files to convert
        output_dir: Directory to save converted files
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf"
        parallel: Whether to process files in parallel (default) or sequentially
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of paths to output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define single file processing function
    def process_single_file(file_path):
        try:
            # Generate output path based on input filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            if storage_format == "hf":
                output_path = os.path.join(output_dir, f"{base_name}_{format_type}_hf")
            else:
                if format_type == "jsonl":
                    output_path = os.path.join(output_dir, f"{base_name}.jsonl")
                else:
                    output_path = os.path.join(output_dir, f"{base_name}_{format_type}.json")
            
            return convert_format(
                input_path=file_path,
                output_path=output_path,
                format_type=format_type,
                config=config,
                storage_format=storage_format
            )
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    output_files = []
    
    # Process files in parallel or sequentially
    if parallel and len(input_files) > 1:
        # Default to CPU count if max_workers not specified
        if max_workers is None:
            import multiprocessing
            max_workers = max(1, multiprocessing.cpu_count())
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_single_file, file_path): file_path 
                             for file_path in input_files}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        output_files.append(result)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    else:
        # Process sequentially
        for file_path in input_files:
            try:
                result = process_single_file(file_path)
                if result:
                    output_files.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    return output_files