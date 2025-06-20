# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# CLI Logic for synthetic-data-kit

import os
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.table import Table

from synthetic_data_kit.core.context import AppContext
from synthetic_data_kit.server.app import run_server
from synthetic_data_kit.utils.config import (
    get_llm_provider,
    get_openai_config,
    get_path_config,
    get_vllm_config,
    load_config,
)

# Initialize Typer app
app = typer.Typer(
    name="synthetic-data-kit",
    help="A toolkit for preparing synthetic datasets for fine-tuning LLMs",
    add_completion=True,
)
console = Console()

# Create app context
ctx = AppContext()


# Define global options
@app.callback()
def callback(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
):
    """
    Global options for the Synthetic Data Kit CLI
    """
    if config:
        ctx.config_path = config
    ctx.config = load_config(ctx.config_path)


@app.command("system-check")
def system_check(
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL to check"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider to check ('vllm' or 'api-endpoint')"
    ),
):
    """
    Check if the selected LLM provider's server is running.
    """
    # Check for API_ENDPOINT_KEY directly from environment
    console.print("Environment variable check:", style="bold blue")
    llama_key = os.environ.get("API_ENDPOINT_KEY")
    console.print(f"API_ENDPOINT_KEY: {'Present' if llama_key else 'Not found'}")
    # Debugging sanity test:
    # if llama_key:
    # console.print(f"  Value starts with: {llama_key[:10]}...")

    # To check the rename bug:
    # console.print("Available environment variables:", style="bold blue")
    # env_vars = [key for key in os.environ.keys() if 'API' in key or 'KEY' in key or 'TOKEN' in key]
    # for var in env_vars:
    #    console.print(f"  {var}")
    # console.print("")
    # Get provider from args or config
    selected_provider = provider or get_llm_provider(ctx.config)

    if selected_provider == "api-endpoint":
        # Get API endpoint config
        api_endpoint_config = get_openai_config(ctx.config)
        api_base = api_base or api_endpoint_config.get("api_base")

        # Check for environment variables
        api_endpoint_key = os.environ.get("API_ENDPOINT_KEY")
        console.print(
            f"API_ENDPOINT_KEY environment variable: {'Found' if api_endpoint_key else 'Not found'}"
        )

        # Set API key with priority: env var > config
        api_key = api_endpoint_key or api_endpoint_config.get("api_key")
        if api_key:
            console.print(
                f"API key source: {'Environment variable' if api_endpoint_key else 'Config file'}"
            )

        model = api_endpoint_config.get("model")

        # Check API endpoint access
        with console.status(f"Checking API endpoint access..."):
            try:
                # Try to import OpenAI
                try:
                    from openai import OpenAI
                except ImportError:
                    console.print("L API endpoint package not installed", style="red")
                    console.print("Install with: pip install openai>=1.0.0", style="yellow")
                    return 1

                # Create client
                client_kwargs = {}
                if api_key:
                    client_kwargs["api_key"] = api_key
                if api_base:
                    client_kwargs["base_url"] = api_base

                # Check API access
                try:
                    client = OpenAI(**client_kwargs)
                    # Try a simple models list request to check connectivity
                    models = client.models.list()
                    console.print(f" API endpoint access confirmed", style="green")
                    if api_base:
                        console.print(f"Using custom API base: {api_base}", style="green")
                    console.print(f"Default model: {model}", style="green")
                    return 0
                except Exception as e:
                    console.print(f"L Error connecting to API endpoint: {str(e)}", style="red")
                    if api_base:
                        console.print(f"Using custom API base: {api_base}", style="yellow")
                    if not api_key and not api_base:
                        console.print(
                            "API key is required. Set in config.yaml or as API_ENDPOINT_KEY env var",
                            style="yellow",
                        )
                    return 1
            except Exception as e:
                console.print(f"L Error: {str(e)}", style="red")
                return 1
    else:
        # Default to vLLM
        # Get vLLM server details
        vllm_config = get_vllm_config(ctx.config)
        api_base = api_base or vllm_config.get("api_base")
        model = vllm_config.get("model")
        port = vllm_config.get("port", 8000)

        with console.status(f"Checking vLLM server at {api_base}..."):
            try:
                response = requests.get(f"{api_base}/models", timeout=2)
                if response.status_code == 200:
                    console.print(f" vLLM server is running at {api_base}", style="green")
                    console.print(f"Available models: {response.json()}")
                    return 0
                else:
                    console.print(f"L vLLM server is not available at {api_base}", style="red")
                    console.print(f"Error: Server returned status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                console.print(f"L vLLM server is not available at {api_base}", style="red")
                console.print(f"Error: {str(e)}")

            # Show instruction to start the server
            console.print("\nTo start the server, run:", style="yellow")
            console.print(f"vllm serve {model} --port {port}", style="bold blue")
            return 1


@app.command()
def ingest(
    input: str = typer.Argument(..., help="File, URL, or directory to parse"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Custom output filename or prefix"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Process files in parallel (default) or sequentially"
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Maximum number of parallel workers (default: CPU count)"
    ),
):
    """
    Parse documents into clean text.

    Accepts a single file, URL, or directory path. When given a directory, will process
    all supported files in that directory regardless of type - perfect for mixed content folders.

    Examples:
    - Single file: synthetic-data-kit ingest data/pdf/paper.pdf
    - Directory with mixed content: synthetic-data-kit ingest data/
    - URL: synthetic-data-kit ingest https://example.com
    - YouTube: synthetic-data-kit ingest https://www.youtube.com/watch?v=dQw4w9WgXcQ

    Supported file types:
    - PDF (.pdf)
    - HTML (.html, .htm)
    - DOCX (.docx)
    - PowerPoint (.pptx)
    - Text (.txt)
    - YouTube URLs
    - Web URLs

    All files will be converted to plain text (.txt) format for further processing.
    """
    from synthetic_data_kit.core.ingest import process_directory, process_file

    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "parsed")

    try:
        # Check if input is a directory
        if os.path.isdir(input):
            from synthetic_data_kit.core.ingest import SUPPORTED_EXTENSIONS

            console.print(f"Processing all supported files in directory: {input}")
            console.print(
                f"Supported file types: {', '.join(ext.upper().lstrip('.') for ext in SUPPORTED_EXTENSIONS)}",
                style="blue",
            )

            with console.status(f"Processing files in {input}..."):
                output_paths = process_directory(
                    input_dir=input,
                    output_dir=output_dir,
                    name_prefix=name,
                    config=ctx.config,
                    parallel=parallel,
                    max_workers=max_workers,
                )

            if output_paths:
                console.print(f" Successfully processed {len(output_paths)} files", style="green")
                if len(output_paths) <= 10:
                    # Show all files if 10 or fewer
                    for path in output_paths:
                        console.print(f" - [bold]{os.path.basename(path)}[/bold]", style="green")
                else:
                    # Show first 5 and last 5 if more than 10
                    for path in output_paths[:5]:
                        console.print(f" - [bold]{os.path.basename(path)}[/bold]", style="green")
                    console.print(
                        f" - ... and {len(output_paths) - 10} more files ...", style="green"
                    )
                    for path in output_paths[-5:]:
                        console.print(f" - [bold]{os.path.basename(path)}[/bold]", style="green")

                console.print(
                    f"\nAll extracted text files are saved to: [bold]{output_dir}[/bold]",
                    style="green",
                )
            else:
                console.print(" No files were successfully processed", style="yellow")
            return 0

        # Single file or URL
        else:
            with console.status(f"Processing {input}..."):
                output_path = process_file(input, output_dir, name, ctx.config)
            console.print(
                f" Text successfully extracted to [bold]{output_path}[/bold]", style="green"
            )
            return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command()
def create(
    input: str = typer.Argument(..., help="File or directory to process"),
    content_type: str = typer.Option(
        "qa", "--type", help="Type of content to generate [qa|summary|cot|cot-enhance]"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    num_pairs: Optional[int] = typer.Option(
        None, "--num-pairs", "-n", help="Target number of QA pairs or CoT examples to generate"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Process files in parallel (default) or sequentially"
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Maximum number of parallel workers (default: CPU count)"
    ),
):
    """
    Generate content from text using local LLM inference.

    Accepts a single file or directory path.
    Examples:
    - Single file: synthetic-data-kit create data/output/paper.txt
    - Directory: synthetic-data-kit create data/output/

    Content types:
    - qa: Generate question-answer pairs from text (use --num-pairs to specify how many)
    - summary: Generate a summary of the text
    - cot: Generate Chain of Thought reasoning examples from text (use --num-pairs to specify how many)
    - cot-enhance: Enhance existing tool-use conversations with Chain of Thought reasoning
      (use --num-pairs to limit the number of conversations to enhance, default is to enhance all)
      (for cot-enhance, the input must be a JSON file with either:
       - A single conversation in 'conversations' field
       - An array of conversation objects, each with a 'conversations' field
       - A direct array of conversation messages)
    """
    from synthetic_data_kit.core.create import process_directory, process_file

    # Check the LLM provider from config
    provider = get_llm_provider(ctx.config)
    console.print(f"L Using {provider} provider", style="green")
    if provider == "api-endpoint":
        # Use API endpoint config
        api_endpoint_config = get_openai_config(ctx.config)
        api_base = api_base or api_endpoint_config.get("api_base")
        model = model or api_endpoint_config.get("model")
        # No server check needed for API endpoint
    else:
        # Use vLLM config
        vllm_config = get_vllm_config(ctx.config)
        api_base = api_base or vllm_config.get("api_base")
        model = model or vllm_config.get("model")

        # Check vLLM server availability
        try:
            response = requests.get(f"{api_base}/models", timeout=2)
            if response.status_code != 200:
                console.print(f"L Error: VLLM server not available at {api_base}", style="red")
                console.print("Please start the VLLM server with:", style="yellow")
                console.print(f"vllm serve {model}", style="bold blue")
                return 1
        except requests.exceptions.RequestException:
            console.print(f"L Error: VLLM server not available at {api_base}", style="red")
            console.print("Please start the VLLM server with:", style="yellow")
            console.print(f"vllm serve {model}", style="bold blue")
            return 1

    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "generated")

    try:
        # Check if input is a directory
        if os.path.isdir(input):
            console.print(f"Processing all text files in directory: {input}")
            with console.status(f"Generating {content_type} content from files in {input}..."):
                output_paths = process_directory(
                    input_dir=input,
                    output_dir=output_dir,
                    config_path=ctx.config_path,
                    api_base=api_base,
                    model=model,
                    content_type=content_type,
                    num_pairs=num_pairs,
                    verbose=verbose,
                    provider=provider,
                    parallel=parallel,
                    max_workers=max_workers,
                )
            console.print(
                f" Successfully generated content for {len(output_paths)} files", style="green"
            )
            for path in output_paths:
                console.print(f" - [bold]{path}[/bold]", style="green")
            return 0

        # Single file
        else:
            with console.status(f"Generating {content_type} content from {input}..."):
                output_path = process_file(
                    input,
                    output_dir,
                    ctx.config_path,
                    api_base,
                    model,
                    content_type,
                    num_pairs,
                    verbose,
                    provider=provider,
                )
            if output_path:
                console.print(f" Content saved to [bold]{output_path}[/bold]", style="green")
            return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command("curate")
def curate(
    input: str = typer.Argument(..., help="Input file or directory to clean"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory or file path"
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t", help="Quality threshold (1-10)"
    ),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Process files in parallel (default) or sequentially"
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Maximum number of parallel workers (default: CPU count)"
    ),
):
    """
    Clean and filter content based on quality.

    Accepts a single file or directory path.
    Examples:
    - Single file: synthetic-data-kit curate data/generated/report_qa_pairs.json
    - Directory: synthetic-data-kit curate data/generated/
    """
    from synthetic_data_kit.core.curate import curate_qa_pairs, process_directory

    # Check the LLM provider from config
    provider = get_llm_provider(ctx.config)

    if provider == "api-endpoint":
        # Use API endpoint config
        api_endpoint_config = get_openai_config(ctx.config)
        api_base = api_base or api_endpoint_config.get("api_base")
        model = model or api_endpoint_config.get("model")
        # No server check needed for API endpoint
    else:
        # Use vLLM config
        vllm_config = get_vllm_config(ctx.config)
        api_base = api_base or vllm_config.get("api_base")
        model = model or vllm_config.get("model")

        # Check vLLM server availability
        try:
            response = requests.get(f"{api_base}/models", timeout=2)
            if response.status_code != 200:
                console.print(f"L Error: VLLM server not available at {api_base}", style="red")
                console.print("Please start the VLLM server with:", style="yellow")
                console.print(f"vllm serve {model}", style="bold blue")
                return 1
        except requests.exceptions.RequestException:
            console.print(f"L Error: VLLM server not available at {api_base}", style="red")
            console.print("Please start the VLLM server with:", style="yellow")
            console.print(f"vllm serve {model}", style="bold blue")
            return 1

    try:
        # Check if input is a directory
        if os.path.isdir(input):
            # Get default output directory if not provided
            if not output:
                output = get_path_config(ctx.config, "output", "cleaned")
                os.makedirs(output, exist_ok=True)

            console.print(f"Processing all QA pairs files in directory: {input}")
            with console.status(f"Cleaning content from files in {input}..."):
                output_paths = process_directory(
                    input_dir=input,
                    output_dir=output,
                    threshold=threshold,
                    api_base=api_base,
                    model=model,
                    config_path=ctx.config_path,
                    verbose=verbose,
                    provider=provider,
                    parallel=parallel,
                    max_workers=max_workers,
                )
            console.print(f" Successfully cleaned {len(output_paths)} files", style="green")
            for path in output_paths:
                console.print(f" - [bold]{path}[/bold]", style="green")
            return 0

        # Single file
        else:
            # Get default output path from config if not provided
            if not output:
                cleaned_dir = get_path_config(ctx.config, "output", "cleaned")
                os.makedirs(cleaned_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(input))[0]
                output = os.path.join(cleaned_dir, f"{base_name}_cleaned.json")

            with console.status(f"Cleaning content from {input}..."):
                result_path = curate_qa_pairs(
                    input,
                    output,
                    threshold,
                    api_base,
                    model,
                    ctx.config_path,
                    verbose,
                    provider=provider,
                )
            console.print(f" Cleaned content saved to [bold]{result_path}[/bold]", style="green")
            return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command("save-as")
def save_as(
    input: str = typer.Argument(..., help="Input file or directory to convert"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Output format [jsonl|alpaca|ft|chatml]"
    ),
    storage: str = typer.Option(
        "json", "--storage", help="Storage format [json|hf]", show_default=True
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory or file path"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Process files in parallel (default) or sequentially"
    ),
    max_workers: Optional[int] = typer.Option(
        None, "--max-workers", help="Maximum number of parallel workers (default: CPU count)"
    ),
):
    """
    Convert to different formats for fine-tuning.

    Accepts a single file or directory path.
    Examples:
    - Single file: synthetic-data-kit save-as data/cleaned/report_cleaned.json
    - Directory: synthetic-data-kit save-as data/cleaned/

    The --format option controls the content format (how the data is structured).
    The --storage option controls how the data is stored (JSON file or HF dataset).

    When using --storage hf, the output will be a directory containing a Hugging Face
    dataset in Arrow format, which is optimized for machine learning workflows.
    """
    from synthetic_data_kit.core.save_as import convert_format, process_directory

    # Get format from args or config
    if not format:
        format_config = ctx.config.get("format", {})
        format = format_config.get("default", "jsonl")

    try:
        # Check if input is a directory
        if os.path.isdir(input):
            # Get default output directory if not provided
            if not output:
                output = get_path_config(ctx.config, "output", "final")
                os.makedirs(output, exist_ok=True)

            console.print(f"Processing all cleaned files in directory: {input}")
            with console.status(
                f"Converting files in {input} to {format} format with {storage} storage..."
            ):
                output_paths = process_directory(
                    input_dir=input,
                    output_dir=output,
                    format_type=format,
                    config=ctx.config,
                    storage_format=storage,
                    parallel=parallel,
                    max_workers=max_workers,
                )
            console.print(f" Successfully converted {len(output_paths)} files", style="green")
            for path in output_paths:
                console.print(f" - [bold]{path}[/bold]", style="green")
            return 0

        # Single file
        else:
            # Set default output path if not provided
            if not output:
                final_dir = get_path_config(ctx.config, "output", "final")
                os.makedirs(final_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(input))[0]

                if storage == "hf":
                    # For HF datasets, use a directory name
                    output = os.path.join(final_dir, f"{base_name}_{format}_hf")
                else:
                    # For JSON files, use appropriate extension
                    if format == "jsonl":
                        output = os.path.join(final_dir, f"{base_name}.jsonl")
                    else:
                        output = os.path.join(final_dir, f"{base_name}_{format}.json")

            with console.status(f"Converting {input} to {format} format with {storage} storage..."):
                output_path = convert_format(
                    input, output, format, ctx.config, storage_format=storage
                )

            if storage == "hf":
                console.print(
                    f" Converted to {format} format and saved as HF dataset to [bold]{output_path}[/bold]",
                    style="green",
                )
            else:
                console.print(
                    f" Converted to {format} format and saved to [bold]{output_path}[/bold]",
                    style="green",
                )
            return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command("server")
def server(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind the server to"),
    port: int = typer.Option(5000, "--port", "-p", help="Port to run the server on"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Run the server in debug mode"),
):
    """
    Start a web interface for the Synthetic Data Kit.

    This launches a web server that provides a UI for all SDK functionality,
    including generating and curating QA pairs, as well as viewing
    and managing generated files.
    """
    provider = get_llm_provider(ctx.config)
    console.print(f"Starting web server with {provider} provider...", style="green")
    console.print(f"Web interface available at: http://{host}:{port}", style="bold green")
    console.print("Press CTRL+C to stop the server.", style="italic")

    # Run the Flask server
    run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
