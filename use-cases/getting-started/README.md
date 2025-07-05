# Getting Started with Synthetic Data Kit: Onboarding Guide

Welcome to the Getting Started guide for Synthetic Data Kit, a comprehensive toolkit for creating high-quality synthetic datasets for fine-tuning Large Language Models.

## Prerequisites

To follow this guide, you'll need:

1. Python 3.8 or later
2. Access to an LLM via local VLLM server
3. The Synthetic Data Kit package (installation instructions below)

## 1. Installation

Install Synthetic Data Kit using pip:

```bash
pip install synthetic-data-kit
```

## 2. Setting Up Directory Structure

Create the necessary directory structure:

```bash
mkdir -p data/{pdf,html,youtube,docx,ppt,txt,output,generated,cleaned,final}
```

## 3. Start VLLM Server (Required)

Synthetic Data Kit requires a running VLLM server. Start one with:

```bash
# If you have the Llama 3 model:
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000

# Alternatively, you can use a smaller model:
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

## 4. Basic Workflow with Examples

Let's go through the process of converting a document to training data:

### Step 1: Verify VLLM Server

First, check if the VLLM server is running:

```bash
synthetic-data-kit system-check
```

You should see a success message if the server is running.

### Step 2: Parse a Document

Parse a PDF document to extract text:

```bash
synthetic-data-kit ingest example_document.pdf
```

This will save the extracted text to `data/output/example_document.txt`.

### Step 3: Generate QA Pairs

Generate question-answer pairs from the parsed text:

```bash
synthetic-data-kit create data/output/example_document.txt
```

This will create QA pairs in `data/generated/example_document_qa_pairs.json`.

### Step 4: Filter for Quality

Filter the generated QA pairs based on quality:

```bash
synthetic-data-kit curate data/generated/example_document_qa_pairs.json
```

This saves the filtered content to `data/cleaned/example_document_cleaned.json`.

### Step 5: Convert to Fine-tuning Format

Convert the cleaned data to a format suitable for fine-tuning:

```bash
synthetic-data-kit save-as data/cleaned/example_document_cleaned.json -f ft
```

The final output will be saved in `data/final/example_document_ft.json` in OpenAI fine-tuning format.

### Step 6: View the Results

You can examine each output to understand the transformation process:

```bash
# View extracted text
cat data/output/example_document.txt | head -n 20

# View generated QA pairs
cat data/generated/example_document_qa_pairs.json | head -n 50

# View filtered pairs
cat data/cleaned/example_document_cleaned.json | head -n 50

# View final fine-tuning format
cat data/final/example_document_ft.json | head -n 50
```

## 5. Command-Line Customization

Synthetic Data Kit supports various command-line options to customize its behavior. Here's how to use them:

### Controlling QA Pair Generation

Generate a specific number of QA pairs:

```bash
synthetic-data-kit create data/output/example_document.txt -n 30
```

Generate Chain of Thought (CoT) reasoning examples instead of QA pairs:

```bash
synthetic-data-kit create data/output/example_document.txt --type cot
```

Enhance tool-use conversations with Chain of Thought reasoning:

```bash
synthetic-data-kit create tool_conversations.json --type cot-enhance
```

## 5.1. Document Processing & Chunking Control

The Synthetic Data Kit automatically handles documents of any size using intelligent processing:

### How Chunking Works

- **Small documents** (< 8000 characters): Processed in a single API call for maximum context
- **Large documents** (≥ 8000 characters): Automatically split into overlapping chunks

### Controlling Chunking with CLI Flags

Use `--chunk-size` and `--chunk-overlap` to customize how large documents are processed:

```bash
# Smaller chunks for detailed processing
synthetic-data-kit create data/output/large_document.txt --chunk-size 2000 --chunk-overlap 100

# Larger chunks for faster processing with more context
synthetic-data-kit create data/output/large_document.txt --chunk-size 6000 --chunk-overlap 300

# Generate many examples with custom chunking
synthetic-data-kit create data/output/large_document.txt --type cot --num-pairs 50 --chunk-size 3000
```

### Understanding Chunking Output

Use `--verbose` to see how your document is being processed:

```bash
synthetic-data-kit create data/output/large_document.txt --type qa --num-pairs 20 --verbose
```

Example output:
```
Generating QA pairs...
Document split into 8 chunks
Using batch size of 32
Processing 8 chunks to generate QA pairs...
  Generated 3 pairs from chunk 1 (total: 3/20)
  Generated 2 pairs from chunk 2 (total: 5/20)
  Generated 3 pairs from chunk 3 (total: 8/20)
  ...
  Reached target of 20 pairs. Stopping processing.
Generated 20 QA pairs total (requested: 20)
```

### Chunking Parameters Guide

| Parameter | Default | Best For | Description |
|-----------|---------|----------|-------------|
| `--chunk-size 2000` | 4000 | Detailed analysis | More chunks, slower but detailed |
| `--chunk-size 6000` | 4000 | Fast processing | Fewer chunks, faster processing |
| `--chunk-overlap 50` | 200 | Reducing repetition | Minimal overlap between chunks |
| `--chunk-overlap 400` | 200 | Preserving context | Maximum context preservation |

### Content Type Consistency

Both QA and CoT generation now use the same chunking logic for consistent behavior:

```bash
# Both will chunk the same way for large documents
synthetic-data-kit create large_doc.txt --type qa --num-pairs 100 --chunk-size 3000
synthetic-data-kit create large_doc.txt --type cot --num-pairs 20 --chunk-size 3000
```

### Customizing Quality Thresholds

Apply a stricter quality threshold during curation:

```bash
synthetic-data-kit curate data/generated/example_document_qa_pairs.json -t 8.5
```

Enable verbose output to see detailed quality ratings:

```bash
synthetic-data-kit curate data/generated/example_document_qa_pairs.json -v
```

### Specifying Output Formats

Convert to ChatML format:

```bash
synthetic-data-kit save-as data/cleaned/example_document_cleaned.json -f chatml
```

Save as a Hugging Face dataset (Arrow format):

```bash
synthetic-data-kit save-as data/cleaned/example_document_cleaned.json -f ft --storage hf
```

## 6. Configuration File Customization

While command-line options are convenient, configuration files provide more extensive customization. Let's create a custom configuration:

### Creating a Custom Configuration

Create a file named `custom_config.yaml` with the following content:

```yaml
# Custom configuration for document processing
vllm:
  api_base: "http://localhost:8000/v1"
  model: "meta-llama/Llama-3.3-70B-Instruct"
  max_retries: 3
  retry_delay: 1.0

generation:
  temperature: 0.5   # Lower temperature for more deterministic outputs
  top_p: 0.95
  
  # Document processing strategy
  processing_strategy: "auto"     # "auto", "single", or "chunking"
  single_call_max_size: 8000      # Documents smaller than this use single call
  
  # Chunking configuration for large documents
  chunk_size: 3000   # Smaller chunks for better processing
  overlap: 300       # More overlap to maintain context
  
  # Generation targets
  num_pairs: 40      # Generate more pairs
  num_cot_examples: 10  # Generate more CoT examples
  
  # Model parameters
  max_tokens: 4096
  batch_size: 32
  
  # Quality settings  
  enable_deduplication: true    # Remove similar questions
  similarity_threshold: 0.8     # How similar is considered duplicate

curate:
  threshold: 8.0     # Higher quality threshold
  batch_size: 16     # Smaller batch size for more detailed processing
  temperature: 0.05  # Lower temperature for more consistent ratings

format:
  default: "ft"      # Default to fine-tuning format
  include_metadata: true
  pretty_json: true

prompts:
  qa_generation: |
    Create {num_pairs} high-quality question-answer pairs about this document.
    
    Focus on questions that:
    1. Test understanding of key concepts
    2. Include important details and examples
    3. Cover main topics comprehensively
    
    Return only the JSON:
    [
      {{
        "question": "Specific question?",
        "answer": "Detailed answer."
      }}
    ]
    
    Text:
    {text}
```

### Using Custom Configuration

Use the custom configuration with any command:

```bash
# Ingest with custom config
synthetic-data-kit -c custom_config.yaml ingest example_document.pdf

# Create with custom config
synthetic-data-kit -c custom_config.yaml create data/output/example_document.txt

# Curate with custom config
synthetic-data-kit -c custom_config.yaml curate data/generated/example_document_qa_pairs.json

# Save with custom config
synthetic-data-kit -c custom_config.yaml save-as data/cleaned/example_document_cleaned.json -f ft
```

## 7. Advanced Command Combinations

You can combine custom configuration with command-line options to override specific settings:

```bash
# Use custom config but override number of pairs
synthetic-data-kit -c custom_config.yaml create data/output/example_document.txt -n 50

# Use custom config but save in a different format
synthetic-data-kit -c custom_config.yaml save-as data/cleaned/example_document_cleaned.json -f chatml
```

## 8. Processing Multiple Documents

For batch processing, you can use a shell script:

```bash
#!/bin/bash
# batch_process.sh

# Process all PDFs in a directory
for file in data/pdf/*.pdf; do
  filename=$(basename "$file" .pdf)
  
  # Full pipeline with custom config
  synthetic-data-kit -c custom_config.yaml ingest "$file"
  synthetic-data-kit -c custom_config.yaml create "data/output/${filename}.txt" -n 20
  synthetic-data-kit -c custom_config.yaml curate "data/generated/${filename}_qa_pairs.json" -t 7.5
  synthetic-data-kit -c custom_config.yaml save-as "data/cleaned/${filename}_cleaned.json" -f ft
done
```

## 9. Customizing Output Location

Specify custom output directories and filenames:

```bash
# Custom output directory for parsed text
synthetic-data-kit ingest example_document.pdf -o custom_output/

# Custom output file for curation
synthetic-data-kit curate data/generated/example_document_qa_pairs.json -o custom_output/high_quality.json
```

## 10. Troubleshooting Common Issues

### Chunking Problems

**Problem**: Getting fewer items than requested
- **Cause**: Document too small or chunks don't contain enough content
- **Solution**: Try smaller `--num-pairs` or combine multiple documents

**Problem**: Repetitive or similar questions
- **Cause**: High chunk overlap causing similar content to be processed multiple times
- **Solution**: Reduce overlap or increase chunk size
```bash
synthetic-data-kit create document.txt --chunk-overlap 50 --chunk-size 5000
```

**Problem**: Poor quality questions across chunks  
- **Cause**: Chunks too small, losing important context
- **Solution**: Increase chunk size to preserve more context
```bash
synthetic-data-kit create document.txt --chunk-size 6000
```

**Problem**: Processing takes too long
- **Cause**: Document creates too many small chunks
- **Solution**: Use larger chunks to reduce processing time
```bash
synthetic-data-kit create document.txt --chunk-size 8000 --num-pairs 20
```

**Problem**: Want to understand what's happening
- **Solution**: Use verbose mode to see chunking details
```bash
synthetic-data-kit create document.txt --verbose
```

### Performance Tips

- **Small documents** (< 8000 chars): Let the tool use single-call processing automatically
- **Medium documents** (8000-50000 chars): Use default settings (`--chunk-size 4000`)
- **Large documents** (> 50000 chars): Consider larger chunks (`--chunk-size 6000-8000`)
- **Very large documents**: Process in smaller batches with fewer `--num-pairs` per run

## Next Steps

For specific use cases and real-world examples, explore the [Use Cases](../Readme.md) section, including:

- Enhancing tool-use conversations with Chain of Thought reasoning
- Creating specialized datasets for different domains
- Advanced customization techniques

After working through this guide, refer to the project's main [README.md](../../ReadMe.MD) and [DOCS.md](../../DOCS.md) for complete documentation of all features and capabilities of the Synthetic Data Kit.