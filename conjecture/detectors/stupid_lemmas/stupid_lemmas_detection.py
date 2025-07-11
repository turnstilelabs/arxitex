import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from conjecture.llms import llms
from conjecture.detectors.stupid_lemmas.stupid_lemmas_detection_models import StupidLemmaDetectionResult
from conjecture.detectors.stupid_lemmas.stupid_lemmas_prompt import StupidLemmaDetectionPromptGenerator


class StupidLemmaDetector:
    def __init__(self):
        pass
    
    def detect_stupid_lemma(self, statement: str) -> StupidLemmaDetectionResult:
        logger.info(f"Detecting stupid lemma for statement: {statement}")
        prompt_generator = StupidLemmaDetectionPromptGenerator()
        prompt = prompt_generator.make_prompt_stupid_lemma(statement)
        
        try:
            return llms.execute_prompt(
                prompt,
                output_class=StupidLemmaDetectionResult,
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            )
        except Exception as e:
            logger.error(f"Error during stupid lemma detection: {e}")
            raise RuntimeError(f"Failed to detect stupid lemma for statement: {statement}")


def process_statements(df: pd.DataFrame, statement_column: str, max_workers: int) -> pd.DataFrame:
    detector = StupidLemmaDetector()
    
    def detect_single_statement(row):
        try:
            result = detector.detect_stupid_lemma(row[statement_column])
            return pd.Series(
                {
                    "is_technical_result": result.is_technical_result,
                    "mathlib_ready": result.mathlib_ready,
                    "technical_result_reason": result.technical_result_reason,
                    "mathlib_reason": result.mathlib_reason,
                    "key_concepts": str(result.key_concepts),
                }
            )
        except Exception as e:
            logger.error(f"Error processing statement: {e}")
            return pd.Series(
                {
                    "is_technical_result": None,
                    "mathlib_ready": None,
                    "technical_result_reason": f"Error: {str(e)}",
                    "mathlib_reason": None,
                    "key_concepts": None,
                }
            )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(detect_single_statement, [row for _, row in df.iterrows()])
        )
    
    results_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, results_df], axis=1)


def detect_lemmas(
    input_file: Path, 
    statement_column: str,
    output_file: Path = None, 
    max_workers: int = 20
) -> pd.DataFrame:
    """
    Process statements from input file and save results to output file.
    """
    logger.info(f"Reading statements from {input_file}")
    
    if input_file.suffix == '.parquet':
        df = pd.read_parquet(input_file)
    elif input_file.suffix == '.csv':
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    total_statements = len(df)
    logger.info(f"Found {total_statements} statements to process")
    
    enhanced_df = process_statements(df, statement_column, max_workers)
    
    successful_detections = enhanced_df["is_technical_result"].notna().sum()
    logger.info(f"Successfully processed {successful_detections} statements")
    
    if output_file:
        if output_file.suffix == '.parquet':
            enhanced_df.to_parquet(output_file)
        else:
            enhanced_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
    
    return enhanced_df


def test_single_statement(statement: str):
    """Test a single statement"""
    detector = StupidLemmaDetector()
    
    try:
        result = detector.detect_stupid_lemma(statement)
        logger.info(f"Statement: {statement}")
        logger.info(f"Is Technical Result: {result.is_technical_result}")
        logger.info(f"Mathlib Ready: {result.mathlib_ready}")
        logger.info(f"Technical Result Reason: {result.technical_result_reason}")
        logger.info(f"Mathlib Reason: {result.mathlib_reason}")
        logger.info(f"Key Concepts: {result.key_concepts}")
    except Exception as e:
        logger.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Detect stupid lemmas")
    parser.add_argument("--statement", type=str, help="Single statement to test")
    parser.add_argument("--input-file", type=Path, help="Input file path")
    parser.add_argument("--output-file", type=Path, help="Output file path")
    parser.add_argument("--statement-column", type=str, help="Column name containing statements")
    parser.add_argument("--max-workers", type=int, default=20, help="Maximum number of workers")
    
    args = parser.parse_args()
    
    if args.statement:
        test_single_statement(args.statement)
    elif args.input_file and args.statement_column:
        detect_lemmas(args.input_file, args.statement_column, args.output_file, args.max_workers)
    else:
        parser.error("Either provide --statement for single test or --input-file and --statement-column for batch processing")


if __name__ == "__main__":
    main()