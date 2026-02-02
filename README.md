# pdf-multicol

`pdf-multicol` is a specialized Python tool using [pdfplumber](https://github.com/jsvine/pdfplumber) to extract text from PDF documents with complex, varying column layouts. It is particularly designed for documents where the number of columns changes within a single page (e.g., a 2-column header followed by a 3-column body), such as RPG rulebooks or academic papers.

## Features

- **Dynamic Layout Detection**: Automatically detects 1, 2, 3, or 4 column layouts for different vertical segments of a single page.
- **Segmented Extraction**: Divides pages into vertical slices to handle layout changes (e.g., mixing single-column headers with multi-column text).
- **Dual Render Modes**:
  - `columns`: Extracts text in reading order (column by column).
  - `table`: Preserves visual row structure (useful for actual tables or lists).
- **Text Cleanup**: Includes heuristics to fix common PDF extraction artifacts like "double-printed" bold text (e.g., fixing "VViirrttuueess").
- **Analysis Tools**: Offers verbose logging and clustering adjustments to fine-tune extraction for difficult documents.

## Requirements

- Python 3.10+
- `pdfplumber`

## Installation

This project is managed with `uv`.

```bash
# Install dependencies
uv sync
```

Alternatively, you can install the dependencies using pip:

```bash
pip install pdfplumber
```

## Usage

The main script is located in `scripts/extract_columns.py`.

### Basic Extraction

Extract text from a PDF, preserving column reading order:

```bash
python scripts/extract_columns.py input.pdf -o output.txt
```

### Table Mode

If the PDF contains tables or you want to preserve the horizontal alignment of text across columns:

```bash
python scripts/extract_columns.py input.pdf --render-mode table -o output.txt
```

### Advanced Configuration

You can fine-tune the extraction parameters if the auto-detection is failing for your specific document:

```bash
# Increase verbosity to see how it's detecting columns
python scripts/extract_columns.py input.pdf -v

# Force a maximum number of columns
python scripts/extract_columns.py input.pdf --max-cols 3

# Adjust the vertical slice height (smaller slices adapt faster to layout changes)
python scripts/extract_columns.py input.pdf --slice-height 100
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `input_pdf` | (Required) | Path to the PDF file to process. |
| `-o`, `--out` | `out.txt` | Path for the output text file. |
| `--render-mode` | `columns` | Output format: `columns` (reading order) or `table` (row-by-row). |
| `-v`, `--verbose` | `False` | Enable detailed logging of the detection process. |
| `--max-cols` | `4` | Maximum number of columns to attempt to detect. |
| `--slice-height` | `120.0` | Height of vertical slices used for layout analysis. |
| `--double-newline` | `False` | Use double newlines between lines in the output. |
| `--show-separators` | `False` | Write segment markers (e.g., `[SEGMENT y=...]`) to the output. |
| `--preview-lines` | `0` | If verbose, print the first N lines of each column to stderr. |

## Project Structure

- `scripts/extract_columns.py`: The core logic for column detection, clustering, and text extraction.
- `assets/`: Contains example inputs and outputs.
- `main.py`: Entry point stub.
