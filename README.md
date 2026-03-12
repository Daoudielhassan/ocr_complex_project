# OCR Complex Project

A complete **Optical Character Recognition (OCR) pipeline** built from classical computer-vision principles — no deep learning required. Characters are described with **HOG (Histogram of Oriented Gradients)** features and classified by a **Support Vector Machine (SVM)**.

The explicit design constraint is **no neural networks, no GPU, no pre-trained weights**. Every component is deterministic, fully explainable, and tunable through plain YAML configuration files.

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

### Stage-by-stage breakdown

#### 1. Preprocessing — `src/preprocessing/`
Cleans and normalises the raw input before any analysis:

| Module | What it does |
|---|---|
| `image_loader.py` | Loads PNG / JPG / TIFF / BMP; optional PDF via `pdf2image` |
| `binarization.py` | Converts to black/white — Otsu, Sauvola (local adaptive), or OpenCV adaptive |
| `denoising.py` | Removes noise — median, Gaussian, bilateral, or NL-means |
| `deskewing.py` | Corrects page rotation by sweeping horizontal projection-profile variance at 0.125° steps |
| `normalization.py` | Rescales to a target height / DPI and adds constant-zero padding |

#### 2. Segmentation — `src/segmentation/`
Locates text regions at decreasing granularity:

```
Full page  →  Layout blocks  →  Lines  →  Words  →  Characters
```

| Module | Algorithm |
|---|---|
| `layout_analyzer.py` | Morphological dilation (30×1 then 1×5 kernels) + connected components |
| `line_segmenter.py` | Horizontal projection profile (row pixel sums) — bridges small gaps |
| `word_segmenter.py` | Vertical projection profile (column pixel sums) |
| `char_segmenter.py` | `cv2.findContours` + bounding-box area/aspect filters, sorted left-to-right |
| `region_filters.py` | Shared filter for min/max area and aspect-ratio constraints |

#### 3. HOG Feature Extraction — `src/features/`
Each character crop is resized to **32×32** and described as a **1764-dimensional HOG vector**:

- 9 gradient orientations, 8×8 pixel cells, 2×2 cell blocks, L2-Hys block normalisation
- A `StandardScaler` then zero-centres and unit-scales the entire feature matrix
- Descriptor shape: $ (\lfloor 32/8 - 1 \rfloor \times 2)^2 \times 9 = 3 \times 3 \times 4 \times 9 = \mathbf{1764} $

#### 4. SVM Classification — `src/classification/`
An **RBF-kernel SVC** maps feature vectors to character class labels:

- Optional `GridSearchCV` over `C` and `gamma`; otherwise 5-fold cross-validation to report CV accuracy before a final full fit
- `LabelManager` wraps `sklearn.LabelEncoder` with pickle save/load
- `SVMModel` wraps `SVC` with save/load and logs the support-vector count

#### 5. Text Reconstruction — `src/reconstruction/`
Assembles predicted characters back into human-readable text:

- **Reading order** — auto-detects columns via x-midpoint gap analysis, then sorts regions by `(column_index, y)`
- **Text rebuilder** — joins chars → words → lines → blocks, separating lines with `\n` and blocks with `\n\n`

#### 6. Post-processing — `src/post_processing/`
Corrects systematic OCR errors:

| Module | Technique |
|---|---|
| `spell_checker.py` | Dictionary lookup via `pyspellchecker`; `@lru_cache` on the SpellChecker instance |
| `language_model.py` | Rule-based heuristics: isolated `l`→`I`, `0`→`O`, broken-hyphenation repair, multiple-space collapse |
| `field_validator.py` | Regex validators for IBAN, ISO/EU dates, currency amounts, email, phone |

#### 7. Export — `src/post_processing/exporter.py`
Writes structured results to **JSON**, **CSV**, **XLSX**, or **Markdown**.

---

## Two Orchestrators

| Class | Location | Purpose |
|---|---|---|
| `TrainingPipeline` | `src/pipeline/training_pipeline.py` | Loads an ImageFolder dataset, fits the scaler, trains the SVM, evaluates on val/test splits, saves all artefacts to `models/` |
| `OCRPipeline` | `src/pipeline/ocr_pipeline.py` | Loads saved artefacts, merges all YAML configs, runs the full inference pipeline on a document image or PDF |

Both are exposed through `main.py` (`--mode train | infer | evaluate`) and through the dedicated scripts in `scripts/`.

---

## Key Design Choices

| Decision | Rationale |
|---|---|
| HOG + SVM, no CNN | Fully explainable, no GPU required, deterministic output |
| All hyperparameters in YAML | Tunable without touching source code |
| `ProjectPaths` singleton | Single source of truth for every directory reference across all modules |
| Synthetic images in tests | Zero external dataset dependency — tests run anywhere, including CI |
| ImageFolder dataset layout | Matches standard ML tooling conventions; compatible with `scripts/prepare_dataset.py` |
| Optional PDF via `pdf2image` | Degrades gracefully with an `ImportError` if Poppler is not installed |

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

## Dataset — Chars74K-Digital-English-Font

The project uses the **Chars74K Digital (Font)** subset of the [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).

| Property | Value |
|---|---|
| Classes | 62 (digits `0–9`, uppercase `A–Z`, lowercase `a–z`) |
| Source folders | `Sample001` – `Sample062` inside `English/Fnt/` |
| Images per class | ~1 016 (synthetic renders from computer fonts) |
| Total images | ~62 992 |
| Image format | PNG, greyscale |

### 1. Download

1. Go to <http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/>
2. Download **EnglishFnt.tgz** (Digital / Font characters)
3. Extract so that the following path exists:

```
data/raw/English/Fnt/
    Sample001/   ← digit "0"
    Sample002/   ← digit "1"
    …
    Sample062/   ← lowercase "z"
```

### 2. Convert to ImageFolder layout

```bash
python scripts/prepare_dataset.py \
    --chars74k-dir data/raw/English/Fnt \
    --out-dir data/dataset \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

This shuffles each class independently and copies images into:

```
data/dataset/
├── train/
│   ├── 0/   ← ~709 images
│   ├── A/   ← ~709 images
│   ├── a/   ← ~709 images
│   └── …    (62 folders)
├── val/
│   └── …    (~15 % per class)
└── test/
    └── …    (~15 % per class)
```

### 3. Train

```bash
python scripts/train_svm.py --data-dir data/dataset
```

### Class–folder mapping

| Chars74K folder | Label | | Chars74K folder | Label |
|---|---|---|---|---|
| Sample001 | `0` | | Sample011 | `A` |
| Sample002 | `1` | | … | … |
| … | … | | Sample036 | `Z` |
| Sample010 | `9` | | Sample037–062 | `a`–`z` |

---

## Running Tests

```bash
pytest
```

All tests run on **synthetically generated NumPy images** — no external dataset required.

---

## License

MIT
