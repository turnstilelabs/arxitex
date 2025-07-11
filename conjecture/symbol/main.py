import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from loguru import logger
from pydantic import TypeAdapter, ValidationError

from conjecture.symbol.document_enhancer import DocumentEnhancer
from conjecture.graph.utils import ArtifactNode

def load_artifacts_from_json(file_path: Path) -> List[ArtifactNode]:
    """Loads artifacts from a JSON file and validates them."""
    if not file_path.exists():
        logger.error(f"Artifact JSON file not found at: {file_path}")
        sys.exit(1)
        
    logger.info(f"Loading artifacts from {file_path}...")
    try:
        # Use Pydantic's TypeAdapter for robust list validation
        ArtifactListAdapter = TypeAdapter(List[ArtifactNode])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Assuming the artifacts are under a "nodes" key
            artifacts = ArtifactListAdapter.validate_python(data.get("nodes", []))
        logger.success(f"Successfully loaded and validated {len(artifacts)} artifacts.")
        return artifacts
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or validate artifacts from {file_path}: {e}")
        sys.exit(1)


def load_latex_content(file_path: Path) -> str:
    """Loads the full LaTeX source code from a file."""
    if not file_path.exists():
        logger.error(f"LaTeX source file not found at: {file_path}")
        sys.exit(1)
    
    logger.info(f"Loading LaTeX source from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.success("LaTeX source loaded.")
    return content

def save_enhanced_artifacts(results: dict, output_path: Path):
    """Saves the enhanced artifact data to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving enhanced artifacts to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved successfully.")


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Enhance mathematical artifacts from a LaTeX paper to make them self-contained.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "json_input",
        type=Path,
        help="Path to the input JSON file containing the extracted artifacts (e.g., paper_artifacts.json)."
    )
    parser.add_argument(
        "latex_input",
        type=Path,
        help="Path to the full LaTeX source file (.tex) for context searching."
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default="output/enhanced_artifacts.json",
        help="Path to save the output JSON file with the enhanced content."
    )

    parser.add_argument(
        "--bank-output-path",
        "-b",
        nargs='?',
        const="output/definition_bank.json",
        default=None,
        type=Path,
        help="Saves the final definition bank. If a path is given, saves there. "
            "If only the flag is present, saves to 'output/definition_bank.json'."
    )
    
    args = parser.parse_args()

    # --- 1. Load Inputs ---
    artifacts = load_artifacts_from_json(args.json_input)
    latex_content = load_latex_content(args.latex_input)

    # --- 2. Run the Enhancement Process ---
    logger.info("Initializing document enhancer...")
    enhancer = DocumentEnhancer(
        artifacts=artifacts,
        latex_content=latex_content
    )
    
    logger.info("Starting artifact enhancement process. This may take some time...")
    enhanced_results = enhancer.run()

    # --- 3. Save the Results ---
    if enhanced_results:
        save_enhanced_artifacts(enhanced_results, args.output_path)
    else:
        logger.warning("Enhancement process finished but produced no results.")

    if args.bank_output_path:
        logger.info(f"Saving definition bank to {args.bank_output_path}...")
        try:
            args.bank_output_path.parent.mkdir(parents=True, exist_ok=True)            
            bank_dict = enhancer.bank.to_dict()

            with open(args.bank_output_path, "w", encoding="utf-8") as f:
                json.dump(bank_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved definition bank to {args.bank_output_path}")

        except Exception as e:
            logger.error(f"Could not save the definition bank: {e}")

if __name__ == "__main__":
    main()