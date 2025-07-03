# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

import concurrent.futures
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from synthetic_data_kit.parsers.docx_parser import DOCXParser
from synthetic_data_kit.parsers.html_parser import HTMLParser

# Import parsers
from synthetic_data_kit.parsers.pdf_parser import PDFParser
from synthetic_data_kit.parsers.ppt_parser import PPTParser
from synthetic_data_kit.parsers.txt_parser import TXTParser
from synthetic_data_kit.parsers.youtube_parser import YouTubeParser

# Define supported extensions and their parsers
EXTENSION_PARSERS = {
    ".pdf": PDFParser,
    ".html": HTMLParser,
    ".htm": HTMLParser,
    ".docx": DOCXParser,
    ".pptx": PPTParser,
    ".txt": TXTParser,
}

# Create a list of supported extensions for file filtering
SUPPORTED_EXTENSIONS = list(EXTENSION_PARSERS.keys())


def determine_parser(file_path: str, config: Dict[str, Any]):
    """Determine the appropriate parser for a file or URL

    Args:
        file_path: Path to the file or URL to parse
        config: Configuration dictionary

    Returns:
        An instance of the appropriate parser

    Raises:
        ValueError: If the file extension is not supported
        FileNotFoundError: If the file does not exist
    """
    # Check if it's a URL
    if file_path.startswith(("http://", "https://")):
        # YouTube URL pattern detection
        if any(youtube_domain in file_path for youtube_domain in ["youtube.com", "youtu.be"]):
            return YouTubeParser()
        # Any other URL is treated as HTML
        return HTMLParser()

    # Handle local file path
    file_path_obj = Path(file_path)

    # Check if file exists
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get extension and check if supported
    ext = file_path_obj.suffix.lower()

    if ext in EXTENSION_PARSERS:
        # Instantiate the appropriate parser
        return EXTENSION_PARSERS[ext]()

    # Extension not supported
    supported_extensions = ", ".join(SUPPORTED_EXTENSIONS)
    raise ValueError(
        f"Unsupported file extension: {ext}. Supported extensions: {supported_extensions}"
    )


def process_file(
    file_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a file using the appropriate parser

    Args:
        file_path: Path to the file or URL to parse
        output_dir: Directory to save parsed text (if None, uses config)
        output_name: Custom filename for output (if None, uses original name)
        config: Configuration dictionary (if None, uses default)

    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Determine parser based on file type
    parser = determine_parser(file_path, config)

    # Parse the file
    content = parser.parse(file_path)

    # Generate output filename if not provided
    if not output_name:
        if file_path.startswith(("http://", "https://")):
            # Extract filename from URL
            if "youtube.com" in file_path or "youtu.be" in file_path:
                # Use video ID for YouTube URLs
                video_id = re.search(r"(?:v=|\.be/)([^&]+)", file_path).group(1)
                output_name = f"youtube_{video_id}.txt"
            else:
                # Use domain for other URLs
                domain = urlparse(file_path).netloc.replace(".", "_")
                output_name = f"{domain}.txt"
        else:
            # Use original filename with .txt extension
            path_obj = Path(file_path)
            output_name = f"{path_obj.stem}.txt"

    # Ensure .txt extension
    if not output_name.endswith(".txt"):
        output_name += ".txt"

    # Save the content
    output_path = output_dir_path / output_name
    parser.save(content, str(output_path))

    return str(output_path)


def process_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    name_prefix: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Process all supported files in a directory

    Automatically detects file types based on extensions and processes each file
    with the appropriate parser.

    Args:
        input_dir: Directory containing files to process
        output_dir: Directory to save parsed text
        name_prefix: Prefix for output filenames
        config: Configuration dictionary
        parallel: Whether to process files in parallel (default) or sequentially
        max_workers: Maximum number of parallel workers

    Returns:
        List of paths to output files
    """
    # Convert input_dir to Path object
    input_path = Path(input_dir)

    # Get all files in the directory
    files = []
    unsupported_files = []

    # Find all files and categorize them
    for file_path in input_path.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(file_path))
            else:
                unsupported_files.append(file_path)

    if not files:
        if unsupported_files:
            # Print info about unsupported files
            print(f"Found {len(unsupported_files)} files, but none with supported extensions.")
            print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
            print(f"Unsupported files: {', '.join(f.name for f in unsupported_files[:5])}")
            if len(unsupported_files) > 5:
                print(f"... and {len(unsupported_files) - 5} more")
        raise ValueError(f"No supported files found in directory: {input_dir}")

    print(f"Found {len(files)} supported files to process")

    return process_multiple_files(
        input_files=files,
        output_dir=output_dir,
        name_prefix=name_prefix,
        config=config,
        parallel=parallel,
        max_workers=max_workers,
    )


def process_multiple_files(
    input_files: List[str],
    output_dir: Optional[str] = None,
    name_prefix: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Process multiple files

    Args:
        input_files: List of files to process
        output_dir: Directory to save parsed text
        name_prefix: Prefix for output filenames
        config: Configuration dictionary
        parallel: Whether to process files in parallel (default) or sequentially
        max_workers: Maximum number of parallel workers

    Returns:
        List of paths to successfully processed output files
    """
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Define single file processing function
    def process_single_file(file_path: str) -> Dict[str, Any]:
        """Process a single file and return result status

        Args:
            file_path: Path to the file to process

        Returns:
            Dictionary with processing status, output path, and error if any
        """
        try:
            # Generate output name if prefix is provided
            output_name = None
            if name_prefix:
                path_obj = Path(file_path)
                output_name = f"{name_prefix}_{path_obj.stem}.txt"

            # Process the file
            result = process_file(file_path, str(output_dir_path), output_name, config)
            return {"status": "success", "path": str(result), "file": file_path}
        except Exception as e:
            return {"status": "error", "error": str(e), "file": file_path}

    output_files: List[str] = []
    failed_files: List[Tuple[str, str]] = []

    # Process files in parallel or sequentially
    if parallel and len(input_files) > 1:
        # Default to CPU count if max_workers not specified
        if max_workers is None:
            import multiprocessing

            max_workers = min(len(input_files), max(1, multiprocessing.cpu_count()))

        # Use ThreadPoolExecutor for I/O-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_file, file_path): file_path
                for file_path in input_files
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        output_files.append(result["path"])
                    else:
                        failed_files.append((file_path, result["error"]))
                except Exception as e:
                    failed_files.append((file_path, str(e)))
    else:
        # Process sequentially
        for file_path in input_files:
            result = process_single_file(file_path)
            if result["status"] == "success":
                output_files.append(result["path"])
            else:
                failed_files.append((file_path, result["error"]))

    # Print summary of processing results
    total_processed = len(output_files) + len(failed_files)
    success_rate = len(output_files) / total_processed if total_processed > 0 else 0

    if total_processed > 0:
        print(f"\nProcessed {len(output_files)} files successfully ({success_rate:.1%})")

        if failed_files:
            print(f"Failed to process {len(failed_files)} files:")
            for file_path, error in failed_files[:5]:
                file_name = Path(file_path).name
                print(f" - {file_name}: {error}")
            if len(failed_files) > 5:
                print(f" - ... and {len(failed_files) - 5} more files")

    return output_files
