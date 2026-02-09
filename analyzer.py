import os
import sys
import logging
import argparse
from utils.extractor import Extractor
from utils.parser import FileParser
from utils.reporter import Reporter
from utils.pdf_to_markdown import pdf_to_markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

def main(input_file, output_dir):
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Starting analysis for: {input_file}")
    
    extractor = Extractor(input_file)
    reporter = Reporter(output_dir)
    workbook_md_paths = []
    
    try:
        # Step 1: Extraction
        logger.info("Extracting files...")
        extract_path, file_list = extractor.extract()
        logger.info(f"Extracted {len(file_list)} files to {extract_path}")

        # Step 2: Analysis
        logger.info("Analyzing files...")
        for file_info in file_list:
            logger.info(f"Processing {file_info['rel_path']}...")
            parser = FileParser(file_info)
            result = parser.parse()
            reporter.add_result(result)
            if file_info["extension"] == ".pdf" and "workbook" in file_info["rel_path"].lower():
                md_name = os.path.splitext(os.path.basename(file_info["rel_path"]))[0] + ".md"
                md_path = os.path.join(output_dir, md_name)
                md_content = pdf_to_markdown(file_info["full_path"])
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                workbook_md_paths.append(md_path)

        # Step 3: Reporting
        logger.info("Generating reports...")
        json_path = reporter.generate_json()
        html_path = reporter.generate_html()
        
        logger.info(f"Analysis complete. Reports generated:")
        logger.info(f"JSON: {json_path}")
        logger.info(f"HTML: {html_path}")
        if workbook_md_paths:
            logger.info("Workbook markdown generated:")
            for path in workbook_md_paths:
                logger.info(f"Markdown: {path}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Cleaning up temporary files...")
        extractor.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze compressed files and generate data overview.")
    parser.add_argument('input_file', help="Path to the compressed file")
    parser.add_argument('--output', default='output', help="Output directory for reports")
    
    args = parser.parse_args()
    
    main(args.input_file, args.output)
