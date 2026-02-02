#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import pdfplumber


# ----------------------------
# Logging
# ----------------------------
def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, file=sys.stderr, flush=True)


# ----------------------------
# Percentile (no numpy)
# ----------------------------
def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("percentile() requires non-empty values")
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if c == f:
        return xs[f]
    d = k - f
    return xs[f] * (1 - d) + xs[c] * d


# ----------------------------
# Title "double-printed" fix
# ----------------------------
def undouble_text(s: str) -> str:
    """
    Fixes strings like "VViirrttuueess" -> "Virtues" (conservative).
    Applies only when the string looks like repeated pairs.
    """
    if len(s) < 6:
        return s
    if len(s) % 2 != 0:
        return s

    pairs_ok = 0
    for i in range(0, len(s), 2):
        if s[i] == s[i + 1]:
            pairs_ok += 1
    ratio = pairs_ok / (len(s) / 2)

    # very conservative threshold
    if ratio >= 0.8:
        return "".join(s[i] for i in range(0, len(s), 2))
    return s


def normalize_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for w in words:
        if "text" in w and isinstance(w["text"], str):
            w["text"] = undouble_text(w["text"])
    return words


# ----------------------------
# Data types
# ----------------------------
@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def clamp(self, w: float, h: float) -> "BBox":
        x0 = max(0.0, min(self.x0, w))
        x1 = max(0.0, min(self.x1, w))
        y0 = max(0.0, min(self.y0, h))
        y1 = max(0.0, min(self.y1, h))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BBox(x0, y0, x1, y1)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


@dataclass
class SliceLayout:
    y0: float
    y1: float
    cols: Optional[List[Tuple[float, float]]]  # None = sparse


@dataclass
class Segment:
    y0: float
    y1: float
    cols: List[Tuple[float, float]]


# ----------------------------
# Extraction / crop
# ----------------------------
def extract_words(page) -> List[Dict[str, Any]]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    return normalize_words(words)


def compute_text_bbox(
    words: Sequence[Dict[str, Any]],
    page_w: float,
    page_h: float,
    p_lo: float,
    p_hi: float,
    pad: float,
) -> Optional[BBox]:
    if not words:
        return None

    x0s = [float(w["x0"]) for w in words]
    x1s = [float(w["x1"]) for w in words]
    tops = [float(w["top"]) for w in words]
    bottoms = [float(w["bottom"]) for w in words]

    x0 = percentile(x0s, p_lo) - pad
    x1 = percentile(x1s, p_hi) + pad
    y0 = percentile(tops, p_lo) - pad
    y1 = percentile(bottoms, p_hi) + pad

    # Anti "comer letras": nunca deixe o bbox mais apertado que min/max +/- pad
    x0 = min(x0, min(x0s) - pad)
    x1 = max(x1, max(x1s) + pad)
    y0 = min(y0, min(tops) - pad)
    y1 = max(y1, max(bottoms) + pad)

    return BBox(x0, y0, x1, y1).clamp(page_w, page_h)


def crop_page(page, bbox: BBox):
    return page.crop((bbox.x0, bbox.y0, bbox.x1, bbox.y1))


# ----------------------------
# Line grouping (use inside a single column OR inside one visual row)
# ----------------------------
def group_words_into_lines(words: Sequence[Dict[str, Any]], y_tol: float) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))

    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_y: Optional[float] = None

    for w in sorted_words:
        y = float(w["top"])
        if current_y is None:
            current_y = y
            current = [w]
            continue

        if abs(y - current_y) <= y_tol:
            current.append(w)
        else:
            current.sort(key=lambda ww: float(ww["x0"]))
            lines.append(current)
            current_y = y
            current = [w]

    if current:
        current.sort(key=lambda ww: float(ww["x0"]))
        lines.append(current)

    return lines


def line_metrics(line_words: Sequence[Dict[str, Any]]) -> Tuple[float, float, float, float, float, str]:
    x0 = min(float(w["x0"]) for w in line_words)
    x1 = max(float(w["x1"]) for w in line_words)
    top = min(float(w["top"]) for w in line_words)
    bottom = max(float(w["bottom"]) for w in line_words)
    x_center = sum((float(w["x0"]) + float(w["x1"])) / 2.0 for w in line_words) / max(1, len(line_words))
    text = " ".join(w["text"] for w in line_words).strip()
    return x0, x1, top, bottom, x_center, text


def detect_fullwidth_lines(
    words: Sequence[Dict[str, Any]],
    y_tol: float,
    text_width: float,
    fullwidth_ratio: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    lines = group_words_into_lines(words, y_tol=y_tol)
    full: List[str] = []
    remaining: List[Dict[str, Any]] = []

    tw = max(1.0, text_width)
    for lw in lines:
        x0, x1, _top, _bottom, _xc, text = line_metrics(lw)
        if text and ((x1 - x0) / tw) >= fullwidth_ratio:
            full.append(text)
        else:
            remaining.extend(lw)

    return full, remaining


# ----------------------------
# Slice / segments
# ----------------------------
def slice_words_by_y(
    words: Sequence[Dict[str, Any]],
    slice_height: float,
    y_min: float,
    y_max: float,
) -> List[Tuple[float, float, List[Dict[str, Any]]]]:
    out: List[Tuple[float, float, List[Dict[str, Any]]]] = []
    y = y_min
    while y < y_max:
        y0 = y
        y1 = min(y_max, y + slice_height)
        bucket: List[Dict[str, Any]] = []
        for w in words:
            top = float(w["top"])
            bottom = float(w["bottom"])
            if bottom >= y0 and top <= y1:
                bucket.append(w)
        out.append((y0, y1, bucket))
        y = y1
    return out


def same_layout(a: List[Tuple[float, float]], b: List[Tuple[float, float]], edge_tol: float) -> bool:
    if len(a) != len(b):
        return False
    for (a0, a1), (b0, b1) in zip(a, b):
        if abs(a0 - b0) > edge_tol or abs(a1 - b1) > edge_tol:
            return False
    return True


def merge_adjacent_slices(
    layouts: List[SliceLayout],
    edge_tol: float,
    min_segment_height: float,
) -> List[Segment]:
    segments: List[Segment] = []
    current: Optional[Segment] = None

    for sl in layouts:
        if sl.cols is None:
            if current is not None:
                current.y1 = sl.y1
            continue

        if current is None:
            current = Segment(y0=sl.y0, y1=sl.y1, cols=list(sl.cols))
            continue

        if same_layout(current.cols, sl.cols, edge_tol=edge_tol):
            current.y1 = sl.y1
        else:
            if (current.y1 - current.y0) >= min_segment_height:
                segments.append(current)
            current = Segment(y0=sl.y0, y1=sl.y1, cols=list(sl.cols))

    if current is not None and (current.y1 - current.y0) >= min_segment_height:
        segments.append(current)

    return segments


# ----------------------------
# Column detection
#   Primary: clustering (k-means 1D on x-centers)
#   Fallback: x-occupancy histogram
# ----------------------------
def kmeans_1d(data: List[float], k: int, iters: int = 40) -> Tuple[List[float], float]:
    data_sorted = sorted(data)
    centers = [data_sorted[int((i + 0.5) / k * (len(data_sorted) - 1))] for i in range(k)]

    for _ in range(iters):
        clusters = [[] for _ in range(k)]
        for x in data:
            idx = min(range(k), key=lambda i: abs(x - centers[i]))
            clusters[idx].append(x)

        new_centers = []
        for i, cl in enumerate(clusters):
            new_centers.append(sum(cl) / len(cl) if cl else centers[i])

        if all(abs(new_centers[i] - centers[i]) < 1e-3 for i in range(k)):
            centers = new_centers
            break
        centers = new_centers

    inertia = 0.0
    for x in data:
        c = min(centers, key=lambda cc: abs(x - cc))
        inertia += (x - c) ** 2

    return centers, inertia


def detect_columns_by_clustering(
    words: Sequence[Dict[str, Any]],
    page_width: float,
    max_cols: int = 4,
    penalty: float = 1.2,
) -> List[Tuple[float, float]]:
    if not words or page_width <= 0:
        return [(0.0, page_width if page_width > 0 else 1.0)]

    xs = [((float(w["x0"]) + float(w["x1"])) / 2.0) for w in words]
    if len(xs) < 10:
        # too few samples: assume 1 col
        return [(0.0, page_width)]

    best_score = float("inf")
    best_centers: Optional[List[float]] = None
    best_k = 1

    for k in range(1, max_cols + 1):
        centers, inertia = kmeans_1d(xs, k)
        score = inertia * (1.0 + penalty * (k - 1))
        if score < best_score:
            best_score = score
            best_centers = centers
            best_k = k

    centers = sorted(best_centers or [page_width / 2.0])

    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(best_k)]
    for w in words:
        xc = (float(w["x0"]) + float(w["x1"])) / 2.0
        idx = min(range(best_k), key=lambda i: abs(xc - centers[i]))
        buckets[idx].append(w)

    cols: List[Tuple[float, float]] = []
    for b in buckets:
        if not b:
            continue
        x0 = min(float(w["x0"]) for w in b)
        x1 = max(float(w["x1"]) for w in b)
        cols.append((x0, x1))

    cols.sort(key=lambda r: r[0])

    # expand slightly
    pad = page_width * 0.01
    cols = [(max(0.0, x0 - pad), min(page_width, x1 + pad)) for (x0, x1) in cols]

    # collapse overlapping/near-overlapping cols (rare but happens)
    merged: List[Tuple[float, float]] = []
    for x0, x1 in cols:
        if not merged:
            merged.append((x0, x1))
        else:
            p0, p1 = merged[-1]
            if x0 <= p1:  # overlap
                merged[-1] = (p0, max(p1, x1))
            else:
                merged.append((x0, x1))

    return merged if merged else [(0.0, page_width)]


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def detect_columns_by_x_occupancy(
    words: Sequence[Dict[str, Any]],
    page_width: float,
    bins: int,
    smooth_window: int,
    threshold_ratio: float,
    min_col_width_ratio: float,
    merge_gap_bins: int,
    max_cols: int,
) -> List[Tuple[float, float]]:
    if not words or page_width <= 0:
        return [(0.0, page_width if page_width > 0 else 1.0)]

    occ = [0.0] * bins

    def to_bin(x: float) -> int:
        return int(max(0, min(bins - 1, (x / page_width) * (bins - 1))))

    for w in words:
        x0 = float(w["x0"])
        x1 = float(w["x1"])
        b0 = to_bin(x0)
        b1 = to_bin(x1)
        if b1 < b0:
            b0, b1 = b1, b0
        for i in range(b0, b1 + 1):
            occ[i] += 1.0

    occ = moving_average(occ, window=smooth_window)
    peak = max(occ) if occ else 0.0
    if peak <= 0:
        return [(0.0, page_width)]

    thresh = peak * threshold_ratio

    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(occ):
        if v >= thresh and not in_seg:
            in_seg = True
            start = i
        elif v < thresh and in_seg:
            in_seg = False
            segments.append((start, i - 1))
    if in_seg:
        segments.append((start, bins - 1))

    if not segments:
        return [(0.0, page_width)]

    merged: List[Tuple[int, int]] = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap_bins:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    min_col_bins = max(1, int(bins * min_col_width_ratio))
    merged = [(s, e) for (s, e) in merged if (e - s + 1) >= min_col_bins]
    if not merged:
        return [(0.0, page_width)]

    if len(merged) > max_cols:
        merged = sorted(merged, key=lambda se: (se[1] - se[0]), reverse=True)[:max_cols]
        merged.sort(key=lambda se: se[0])

    def bin_to_x(i: int) -> float:
        return (i / (bins - 1)) * page_width

    cols = [(bin_to_x(s), bin_to_x(e)) for (s, e) in merged]
    pad = page_width * 0.005
    cols = [(max(0.0, x0 - pad), min(page_width, x1 + pad)) for (x0, x1) in cols]
    return cols if cols else [(0.0, page_width)]


def detect_columns(
    words: Sequence[Dict[str, Any]],
    page_width: float,
    args,
    verbose: bool,
    context: str = "",
) -> List[Tuple[float, float]]:
    # primary
    cols = detect_columns_by_clustering(words, page_width=page_width, max_cols=args.max_cols, penalty=args.cluster_penalty)

    # sanity check: if clustering returns 1 column but we likely have multi-col,
    # try occupancy and pick the one with more columns.
    occ_cols = detect_columns_by_x_occupancy(
        words,
        page_width=page_width,
        bins=args.bins,
        smooth_window=args.smooth_window,
        threshold_ratio=args.threshold_ratio,
        min_col_width_ratio=args.min_col_width_ratio,
        merge_gap_bins=args.merge_gap_bins,
        max_cols=args.max_cols,
    )

    # Choose:
    # - prefer the one with MORE columns (when it makes sense),
    # - but never exceed max_cols
    chosen = cols
    if len(occ_cols) > len(cols):
        chosen = occ_cols

    if verbose:
        log(f"    [detect_columns{(' '+context) if context else ''}] clustering={len(cols)} {[(round(a,1), round(b,1)) for a,b in cols]}", True)
        log(f"    [detect_columns{(' '+context) if context else ''}] occupancy ={len(occ_cols)} {[(round(a,1), round(b,1)) for a,b in occ_cols]}", True)
        log(f"    [detect_columns{(' '+context) if context else ''}] chosen   ={len(chosen)} {[(round(a,1), round(b,1)) for a,b in chosen]}", True)

    return chosen if chosen else [(0.0, page_width)]


# ----------------------------
# Rendering helpers
# ----------------------------
def split_words_into_columns(words: Sequence[Dict[str, Any]], cols: List[Tuple[float, float]]) -> List[List[Dict[str, Any]]]:
    if not cols:
        return [list(words)]

    buckets: List[List[Dict[str, Any]]] = [[] for _ in cols]
    centers = [((x0 + x1) / 2.0) for (x0, x1) in cols]

    for w in words:
        xc = (float(w["x0"]) + float(w["x1"])) / 2.0
        chosen = None
        for i, (x0, x1) in enumerate(cols):
            if x0 <= xc <= x1:
                chosen = i
                break
        if chosen is None:
            dists = [abs(xc - c) for c in centers]
            chosen = int(min(range(len(cols)), key=lambda i: dists[i]))
        buckets[chosen].append(w)

    return buckets


def render_segment_as_columns(
    all_words: Sequence[Dict[str, Any]],
    seg: Segment,
    y_tol: float,
    verbose: bool,
    preview_lines: int,
) -> List[List[str]]:
    ws = [w for w in all_words if float(w["bottom"]) >= seg.y0 and float(w["top"]) <= seg.y1]
    col_words = split_words_into_columns(ws, seg.cols)

    col_lines_text: List[List[str]] = []
    for col_idx, bucket in enumerate(col_words):
        lines = group_words_into_lines(bucket, y_tol=y_tol)
        rendered: List[Tuple[float, str]] = []
        for lw in lines:
            _x0, _x1, top, _bottom, _xc, text = line_metrics(lw)
            if text:
                rendered.append((top, text))
        rendered.sort(key=lambda t: t[0])
        texts = [t[1] for t in rendered]
        col_lines_text.append(texts)

        if verbose and preview_lines > 0:
            log(f"    [segment y={seg.y0:.0f}-{seg.y1:.0f}] col {col_idx+1} lines={len(texts)}", verbose)
            for i, t in enumerate(texts[:preview_lines], start=1):
                log(f"      [preview col {col_idx+1} line {i}] {t}", verbose)

    return col_lines_text


def render_segment_as_table_rows(
    all_words: Sequence[Dict[str, Any]],
    seg: Segment,
    y_tol: float,
    cols: List[Tuple[float, float]],
    cell_sep: str,
) -> List[str]:
    ws = [w for w in all_words if float(w["bottom"]) >= seg.y0 and float(w["top"]) <= seg.y1]
    if not ws:
        return []

    lines = group_words_into_lines(ws, y_tol=y_tol)

    out_lines: List[Tuple[float, str]] = []
    for lw in lines:
        buckets = split_words_into_columns(lw, cols)

        cells: List[str] = []
        for b in buckets:
            if not b:
                cells.append("")
                continue
            b_sorted = sorted(b, key=lambda w: float(w["x0"]))
            cell_text = " ".join(w["text"] for w in b_sorted).strip()
            cells.append(cell_text)

        while cells and cells[-1] == "":
            cells.pop()

        line_text = cell_sep.join(cells).rstrip()
        if line_text.strip():
            top = min(float(w["top"]) for w in lw)
            out_lines.append((top, line_text))

    out_lines.sort(key=lambda t: t[0])
    return [t[1] for t in out_lines]


# ----------------------------
# Output writers
# ----------------------------
def write_output_columns(
    out,
    page_no: int,
    fullwidth_lines: List[str],
    segments_cols_lines: List[Tuple[Segment, List[List[str]]]],
    show_separators: bool,
    double_newline: bool,
    show_column_headers: bool,
):
    nl = "\n\n" if double_newline else "\n"
    out.write(f"\n\n=== PAGE {page_no} ===\n\n")

    # Put full-width lines at start of Column 1
    if segments_cols_lines and fullwidth_lines:
        first_seg, cols_lines = segments_cols_lines[0]
        if cols_lines:
            cols_lines[0] = fullwidth_lines + cols_lines[0]
            segments_cols_lines[0] = (first_seg, cols_lines)

    for seg, cols_lines in segments_cols_lines:
        if show_separators:
            out.write(f"[SEGMENT y={seg.y0:.0f}-{seg.y1:.0f} cols={len(seg.cols)}]\n\n")

        for c_idx, lines in enumerate(cols_lines, start=1):
            if show_column_headers:
                out.write(f"-- Column {c_idx} --\n\n")
            for line in lines:
                out.write(line + nl)


def write_output_table(
    out,
    page_no: int,
    fullwidth_lines: List[str],
    segments_table_lines: List[Tuple[Segment, List[str]]],
    double_newline: bool,
    show_separators: bool,
):
    nl = "\n\n" if double_newline else "\n"
    out.write(f"\n\n=== PAGE {page_no} ===\n\n")

    for line in fullwidth_lines:
        out.write(line + nl)

    for seg, lines in segments_table_lines:
        if show_separators:
            out.write(f"[SEGMENT y={seg.y0:.0f}-{seg.y1:.0f} cols={len(seg.cols)}]\n\n")
        for line in lines:
            out.write(line + nl)


# ----------------------------
# Page pipeline
# ----------------------------
def process_page(page, page_no: int, args, out):
    verbose = args.verbose

    words = extract_words(page)
    log(f"[page {page_no}] raw_words={len(words)} w={page.width:.1f} h={page.height:.1f}", verbose)

    if not words:
        out.write(f"\n\n=== PAGE {page_no} ===\n\n(no text extracted)\n")
        print(f"[page {page_no}] WARNING: no words extracted (consider OCR fallback)", file=sys.stderr, flush=True)
        return

    # Crop robust
    bbox = compute_text_bbox(
        words,
        page_w=page.width,
        page_h=page.height,
        p_lo=args.crop_p_lo,
        p_hi=args.crop_p_hi,
        pad=args.crop_pad,
    )
    if bbox:
        log(f"[page {page_no}] bbox={bbox}", verbose)
        page_c = crop_page(page, bbox)
        words_c = extract_words(page_c)
        log(f"[page {page_no}] after_crop_words={len(words_c)}", verbose)
        if words_c:
            page = page_c
            words = words_c

    if not words:
        out.write(f"\n\n=== PAGE {page_no} ===\n\n(no text extracted after crop)\n")
        print(f"[page {page_no}] WARNING: no words after crop (consider OCR)", file=sys.stderr, flush=True)
        return

    # Full-width lines
    tb = compute_text_bbox(words, page.width, page.height, p_lo=0.5, p_hi=99.5, pad=0.0)
    text_w = tb.width if tb else page.width

    fullwidth_lines, remaining = detect_fullwidth_lines(
        words,
        y_tol=args.y_tol,
        text_width=text_w,
        fullwidth_ratio=args.fullwidth_ratio,
    )
    log(f"[page {page_no}] fullwidth_lines={len(fullwidth_lines)} remaining_words={len(remaining)}", verbose)

    # Slices
    slices = slice_words_by_y(remaining, args.slice_height, 0.0, page.height)
    layouts: List[SliceLayout] = []

    for idx, (y0, y1, ws) in enumerate(slices):
        if len(ws) < args.min_words_per_slice:
            layouts.append(SliceLayout(y0=y0, y1=y1, cols=None))
            log(f"[page {page_no}] slice {idx} y=[{y0:.0f},{y1:.0f}] words={len(ws)} layout=None", verbose)
            continue

        cols = detect_columns(ws, page_width=page.width, args=args, verbose=verbose, context=f"page{page_no}:slice{idx}")
        layouts.append(SliceLayout(y0=y0, y1=y1, cols=cols))
        log(f"[page {page_no}] slice {idx} y=[{y0:.0f},{y1:.0f}] words={len(ws)} cols={len(cols)} {[(round(a,1), round(b,1)) for a,b in cols]}", verbose)

    # Merge slices -> segments
    segments = merge_adjacent_slices(layouts, edge_tol=args.edge_tol, min_segment_height=args.min_segment_height)
    log(f"[page {page_no}] segments={len(segments)}", verbose)

    # Fallback segment if needed
    if not segments:
        cols = detect_columns(remaining, page_width=page.width, args=args, verbose=verbose, context=f"page{page_no}:fallback")
        segments = [Segment(y0=0.0, y1=page.height, cols=cols)]
        log(f"[page {page_no}] fallback segment cols={len(cols)}", verbose)

    # Render
    if args.render_mode == "table":
        segments_table_lines: List[Tuple[Segment, List[str]]] = []
        for si, seg in enumerate(segments):
            log(f"[page {page_no}] render(table) segment {si} y=[{seg.y0:.0f},{seg.y1:.0f}] cols={len(seg.cols)}", verbose)
            lines = render_segment_as_table_rows(
                remaining,
                seg=seg,
                y_tol=args.y_tol,
                cols=seg.cols,
                cell_sep=args.cell_sep,
            )
            segments_table_lines.append((seg, lines))

        write_output_table(
            out=out,
            page_no=page_no,
            fullwidth_lines=fullwidth_lines,
            segments_table_lines=segments_table_lines,
            double_newline=args.double_newline,
            show_separators=args.show_separators,
        )
    else:
        segments_cols_lines: List[Tuple[Segment, List[List[str]]]] = []
        for si, seg in enumerate(segments):
            log(f"[page {page_no}] render(columns) segment {si} y=[{seg.y0:.0f},{seg.y1:.0f}] cols={len(seg.cols)}", verbose)
            cols_lines = render_segment_as_columns(
                remaining,
                seg=seg,
                y_tol=args.y_tol,
                verbose=verbose,
                preview_lines=args.preview_lines,
            )
            segments_cols_lines.append((seg, cols_lines))

        write_output_columns(
            out=out,
            page_no=page_no,
            fullwidth_lines=fullwidth_lines,
            segments_cols_lines=segments_cols_lines,
            show_separators=args.show_separators,
            double_newline=args.double_newline,
            show_column_headers=not args.no_column_headers,
        )


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract multi-column PDF text with per-segment layouts (1-4 cols).")

    ap.add_argument("pdf", help="Input PDF path")
    ap.add_argument("-o", "--out", default="out.txt", help="Output text file path")

    ap.add_argument("--render-mode", choices=["columns", "table"], default="columns",
                    help="columns = coluna inteira (listas). table = linha visual por linha (tabelas).")
    ap.add_argument("--cell-sep", default="\t", help="Separador entre células no modo table (default: TAB).")

    ap.add_argument("--y-tol", type=float, default=2.0)
    ap.add_argument("--slice-height", type=float, default=120.0)
    ap.add_argument("--min-words-per-slice", type=int, default=20)

    ap.add_argument("--fullwidth-ratio", type=float, default=0.80)

    # detector antigo (fallback)
    ap.add_argument("--bins", type=int, default=240)
    ap.add_argument("--smooth-window", type=int, default=9)
    ap.add_argument("--threshold-ratio", type=float, default=0.18)
    ap.add_argument("--min-col-width-ratio", type=float, default=0.10)
    ap.add_argument("--merge-gap-bins", type=int, default=6)

    # detector novo (clustering)
    ap.add_argument("--cluster-penalty", type=float, default=1.2, help="Penalty for higher k in clustering detector.")
    ap.add_argument("--max-cols", type=int, default=4)

    ap.add_argument("--edge-tol", type=float, default=20.0)
    ap.add_argument("--min-segment-height", type=float, default=40.0)

    # Crop (safe defaults)
    ap.add_argument("--crop-p-lo", type=float, default=0.5)
    ap.add_argument("--crop-p-hi", type=float, default=99.5)
    ap.add_argument("--crop-pad", type=float, default=18.0)

    ap.add_argument("--double-newline", action="store_true")
    ap.add_argument("--show-separators", action="store_true")
    ap.add_argument("--no-column-headers", action="store_true")

    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--preview-lines", type=int, default=0)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    with pdfplumber.open(args.pdf) as pdf, open(args.out, "w", encoding="utf-8") as out:
        for page_no, page in enumerate(pdf.pages, start=1):
            process_page(page, page_no, args, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
