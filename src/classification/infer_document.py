"""Full document OCR inference: image → structured text result."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logger import get_logger
from src.utils.paths import PATHS

log = get_logger(__name__)


def infer_document(
    img_path: str | Path,
    config: dict[str, Any],
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    encoder_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete OCR inference pipeline on one document.

    Parameters
    ----------
    img_path:
        Path to the input file (PNG, JPG, BMP, or PDF).
    config:
        Merged configuration dictionary (from YAML files).
    model_path, scaler_path, encoder_path:
        Override paths for model artefacts.  Defaults from :data:`PATHS`.

    Returns
    -------
    dict
        ``{ "text": str, "blocks": int, "source": str }``
    """
    # ── lazy imports to keep startup time low ─────────────────
    from src.preprocessing.image_loader import load_document
    from src.preprocessing.binarization import binarize
    from src.preprocessing.denoising import denoise
    from src.preprocessing.deskewing import deskew
    from src.preprocessing.normalization import normalize
    from src.segmentation.layout_analyzer import analyze_layout
    from src.segmentation.line_segmenter import segment_lines
    from src.segmentation.word_segmenter import segment_words
    from src.segmentation.char_segmenter import segment_chars
    from src.classification.model import SVMModel
    from src.classification.label_encoder import LabelManager
    from src.features.scaling import load_scaler
    from src.classification.predict import predict_chars
    from src.reconstruction.text_rebuilder import rebuild_text
    from src.reconstruction.reading_order import sort_reading_order

    # ── resolve artefact paths ─────────────────────────────────
    model_path = model_path or PATHS.svm_model
    scaler_path = scaler_path or PATHS.scaler
    encoder_path = encoder_path or PATHS.label_encoder

    model = SVMModel.load(model_path)
    scaler = load_scaler(scaler_path)
    label_mgr = LabelManager.load(encoder_path)

    # ── preprocessing ─────────────────────────────────────────
    pp = config.get("preprocessing", {})
    img = load_document(img_path)
    img = denoise(img, method=pp.get("denoise_method", "median"))
    if pp.get("deskew", True):
        img = deskew(img, max_angle=float(pp.get("deskew_max_angle", 15.0)))
    img = binarize(img, method=pp.get("binarize_method", "otsu"))
    img = normalize(
        img,
        target_height=int(pp.get("target_height", 1024)),
        padding=int(pp.get("padding", 0)),
    )

    # ── segmentation ──────────────────────────────────────────
    seg = config.get("segmentation", {})
    blocks = analyze_layout(img, min_block_area=int(seg.get("min_block_area", 500)))
    if not blocks:
        h, w = img.shape
        blocks = [(0, 0, w, h)]
    blocks = sort_reading_order(blocks)

    # ── feature / prediction parameters ──────────────────────
    feat = config.get("features", {})
    resize_to = tuple(feat.get("resize_to", [32, 32]))
    orientations = int(feat.get("orientations", 9))
    pixels_per_cell = tuple(feat.get("pixels_per_cell", [8, 8]))
    cells_per_block = tuple(feat.get("cells_per_block", [2, 2]))

    # ── prediction loop ───────────────────────────────────────
    all_char_predictions: list[list[list[str]]] = []
    for bx, by, bw, bh in blocks:
        block_img = img[by : by + bh, bx : bx + bw]
        line_imgs, _ = segment_lines(
            block_img,
            min_line_height=int(seg.get("min_line_height", 5)),
            gap_threshold=int(seg.get("line_gap_threshold", 3)),
        )
        block_lines: list[list[str]] = []
        for line_img in line_imgs:
            word_imgs, _ = segment_words(
                line_img,
                min_word_width=int(seg.get("min_word_width", 3)),
                gap_threshold=int(seg.get("word_gap_threshold", 5)),
            )
            line_chars: list[str] = []
            for word_img in word_imgs:
                char_imgs, _ = segment_chars(
                    word_img,
                    min_char_area=int(seg.get("min_char_area", 5)),
                )
                if char_imgs:
                    preds = predict_chars(
                        char_imgs, model, scaler, label_mgr,
                        resize_to=resize_to,
                        orientations=orientations,
                        pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block,
                    )
                    line_chars.extend(preds)
                    line_chars.append(" ")
            block_lines.append(line_chars)
        all_char_predictions.append(block_lines)

    text = rebuild_text(all_char_predictions)
    log.info("infer_document: %d blocks, %d chars", len(blocks), len(text))
    return {
        "text": text,
        "blocks": len(all_char_predictions),
        "source": str(img_path),
    }
