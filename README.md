# OCR Complex Project

A complete **Optical Character Recognition (OCR) pipeline** built from classical computer-vision principles — no deep learning required. Characters are described with **HOG (Histogram of Oriented Gradients)** features and classified by a **Support Vector Machine (SVM)**.

---

## Pipeline Architecture

```
Document image
      │
      ▼
┌─────────────────────┐
│   Preprocessing     │  binarize · denoise · deskew · normalize
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Segmentation      │  layout → lines → words → characters
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  HOG Feature Extr.  │  32×32 patch → 1764-dim descriptor
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   SVM Classifier    │  RBF kernel · StandardScaler
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Text Reconstruction│  reading order · word / line / block assembly
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Post-processing    │  spell check · LM rules · field validation
└─────────┬───────────┘
          │
          ▼
     JSON / CSV / XLSX / Markdown export
```

---

## Project Structure

```
ocr_complex_project/
├── configs/          # YAML configuration files
├── data/             # Raw, interim, and annotated data
├── models/           # Saved SVM model, scaler, label encoder
├── artifacts/        # Confusion matrices, error reports, etc.
├── runs/             # Timestamped experiment logs
├── scripts/          # Standalone CLI helpers
├── src/              # Core library
│   ├── utils/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── features/
│   ├── classification/
│   ├── reconstruction/
│   ├── evaluation/
│   ├── post_processing/
│   └── pipeline/
├── tests/            # pytest suite (synthetic images, no external data)
├── notebooks/        # Exploratory Jupyter notebooks
├── main.py           # Unified CLI entry-point
└── requirements.txt
```

---

## Installation

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ocr_complex_project
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Or** install as an editable package with dev extras:

```bash
pip install -e ".[dev]"
```

**Optional PDF support:**

```bash
pip install -e ".[pdf]"
```

---

## Quick Start

### Train a model

```bash
python main.py --mode train --data-dir data/dataset --config-dir configs
```

Or use the dedicated script:

```bash
python scripts/train_svm.py --data-dir data/dataset
```

### Run OCR on a single image

```bash
python main.py --mode infer --input path/to/image.png
```

### OCR an entire folder

```bash
python scripts/run_ocr_on_folder.py --input-dir path/to/images --output-dir results/
```

### Evaluate on a test split

```bash
python main.py --mode evaluate --data-dir data/dataset
# or
python scripts/evaluate_model.py --data-dir data/dataset
```

---

## Configuration

All behaviour is controlled by YAML files in `configs/`. Interesting knobs:

| File | Key settings |
|------|-------------|
| `preprocessing.yaml` | `binarize_method` (`otsu` / `sauvola` / `adaptive`), `denoise_method`, `target_height` |
| `segmentation.yaml` | `min_block_area`, `line_gap_threshold`, `word_gap_threshold`, `min_char_area` |
| `features.yaml` | `resize_to`, `orientations`, `pixels_per_cell`, `cells_per_block` |
| `svm.yaml` | `kernel`, `C`, `gamma`, `grid_search`, `cv_folds` |
| `export.yaml` | `format` (`json` / `csv` / `xlsx` / `markdown`) |

---

## Dataset Layout

The training pipeline expects an **ImageFolder** layout:

```
data/dataset/
├── train/
│   ├── A/  (one folder per class label)
│   │   ├── 001.png
│   │   └── …
│   └── …
├── val/
└── test/
```

Use `scripts/prepare_dataset.py` to automatically segment a labelled document collection into this layout.

---

## Running Tests

```bash
pytest
```

All tests run on **synthetically generated NumPy images** — no external dataset required.

---

## License

MIT
