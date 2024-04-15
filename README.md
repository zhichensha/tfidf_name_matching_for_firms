
# Fuzzy Match Python Script

## Overview
The `fuzzy_match.py` script provides comprehensive functionality for fuzzy matching of company names across datasets, utilizing advanced text processing, TF-IDF vectorization, and nearest neighbor search.

## Features
- **Text Preprocessing**: Cleans and normalizes text, handles non-ASCII characters, abbreviations, and removes specific stopwords.
- **Vectorization and N-Gram Analysis**: Uses TF-IDF vectorization and custom n-gram analysis to prepare text data for matching.
- **Fuzzy Matching**: Implements efficient fuzzy matching using the `nmslib` library, ideal for comparing textual data across different datasets.
- **Yearly and Quarterly Matching**: Facilitates temporal segmentation in data matching, essential for tasks that require historical data alignment.

## Usage
This script is suitable for data merging tasks such as aligning customer databases or linking financial records where company names may have discrepancies.

## Requirements
- Python 3.x
- Libraries: `pandas`, `sklearn`, `nmslib`, `ftfy`

## Installation
Ensure you have Python installed and then install required packages using:
```bash
pip install pandas sklearn nmslib ftfy
```

## Example
Ideal for financial institutions aiming to consolidate customer or transaction data across multiple databases. See the  `create_dealscan_with_compustat_callreport_tfidf_meausres.py` for example use.

## License
Specify the license under which this script is released, ensuring users know their rights for using and modifying the software.
