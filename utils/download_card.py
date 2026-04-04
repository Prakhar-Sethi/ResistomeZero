"""
Download CARD (Comprehensive Antibiotic Resistance Database) data.

CARD provides multiple downloadable files:
- card.json: Complete CARD database in JSON format
- aro.obo: Antibiotic Resistance Ontology in OBO format
- nucleotide_fasta_protein_homolog_model.fasta: Sequences
- protein_fasta_protein_homolog_model.fasta: Protein sequences
- snps.txt: SNP data
"""

import os
import json
import gzip
import bz2
import requests
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "card"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CARD download URLs (using latest version)
CARD_BASE_URL = "https://card.mcmaster.ca/latest/data"
FILES_TO_DOWNLOAD = {
    "card.json": f"{CARD_BASE_URL}",  # Main database
    "aro.obo": "https://card.mcmaster.ca/latest/ontology",  # Ontology
    "aro_index.tsv": "https://card.mcmaster.ca/latest/index",  # Index file
}

# Additional sequences (optional - can be large)
SEQUENCE_FILES = {
    "nucleotide_fasta_protein_homolog_model.fasta":
        "https://card.mcmaster.ca/latest/variants",
    "protein_fasta_protein_homolog_model.fasta":
        "https://card.mcmaster.ca/latest/sequences",
}


def download_file(url: str, filepath: Path, description: str = "Downloading") -> bool:
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        filepath: Path to save file
        description: Description for progress bar

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {description} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        logger.info(f"Successfully downloaded to {filepath}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def explore_card_json(json_path: Path):
    """
    Explore the structure of CARD JSON file.

    Args:
        json_path: Path to card.json file
    """
    logger.info("Exploring CARD JSON structure...")

    # Try different compression formats
    card_data = None

    # Try bz2
    try:
        with bz2.open(json_path, 'rt', encoding='utf-8') as f:
            card_data = json.load(f)
        logger.info("File was bz2 compressed, decompressed successfully")
    except Exception:
        pass

    # Try gzip
    if card_data is None:
        try:
            with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                card_data = json.load(f)
            logger.info("File was gzipped, decompressed successfully")
        except Exception:
            pass

    # Try plain text
    if card_data is None:
        with open(json_path, 'r', encoding='utf-8') as f:
            card_data = json.load(f)
        logger.info("File was plain JSON")

    logger.info(f"CARD JSON keys: {list(card_data.keys())}")

    # Explore structure
    if isinstance(card_data, dict):
        for key, value in card_data.items():
            if isinstance(value, dict):
                logger.info(f"\n{key}: {len(value)} entries")
                # Show first entry as example
                if value:
                    first_key = list(value.keys())[0]
                    logger.info(f"  Example key: {first_key}")
                    logger.info(f"  Example entry keys: {list(value[first_key].keys())[:10]}")
            elif isinstance(value, list):
                logger.info(f"\n{key}: list with {len(value)} items")
            else:
                logger.info(f"\n{key}: {type(value).__name__}")

    # Save summary
    summary_path = json_path.parent / "card_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CARD Database Summary\n")
        f.write("=" * 50 + "\n\n")

        if isinstance(card_data, dict):
            for key in card_data.keys():
                if isinstance(card_data[key], dict):
                    f.write(f"{key}: {len(card_data[key])} entries\n")
                elif isinstance(card_data[key], list):
                    f.write(f"{key}: list with {len(card_data[key])} items\n")
                else:
                    f.write(f"{key}: {card_data[key]}\n")

    logger.info(f"Summary saved to {summary_path}")


def main():
    """Main function to download CARD data."""
    logger.info("Starting CARD data download...")
    logger.info(f"Download directory: {DATA_DIR}")

    # Download main files
    for filename, url in FILES_TO_DOWNLOAD.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            logger.info(f"{filename} already exists, skipping...")
            continue
        download_file(url, filepath, filename)

    # Optionally download sequence files (can be large)
    # For now, skip sequence files to save time and space
    # Uncomment below to download:
    # for filename, url in SEQUENCE_FILES.items():
    #     filepath = DATA_DIR / filename
    #     if filepath.exists():
    #         logger.info(f"{filename} already exists, skipping...")
    #         continue
    #     download_file(url, filepath, filename)
    logger.info("\nSkipping large sequence files for now...")

    # Explore CARD JSON if downloaded
    card_json_path = DATA_DIR / "card.json"
    if card_json_path.exists():
        explore_card_json(card_json_path)

    logger.info("\n" + "=" * 50)
    logger.info("Download complete!")
    logger.info(f"Files saved to: {DATA_DIR}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
