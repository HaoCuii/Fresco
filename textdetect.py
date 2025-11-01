import io
import json
import logging
import math
import os
import shutil

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(message)s")
PASS2_LOG = logging.getLogger("ocr.pass2")
OCR_LOG = logging.getLogger("ocr.compare")

# ---------------------------------------------------------------------
# tesseract setup
# ---------------------------------------------------------------------
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd = win_path
    else:
        print("Tesseract OCR not found. Install it and put it on PATH.")
        raise SystemExit(1)

# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
DPI = 300
MIN_CONF = 80          # regular OCR detections
SHAPE_MIN_CONF = 30    # shape OCR detections
SAVE_DEBUG_IMAGES = False


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _tess_image_to_data(img, config):
    """Run tesseract with given config and return items above MIN_CONF."""
    data = pytesseract.image_to_data(
        img,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    items = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1

        if text and conf > MIN_CONF:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            items.append(
                {"text": text, "bbox": (x, y, w, h), "conf": conf}
            )
    return items


def _boxes_overlap(b1, b2, threshold=0.5):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return False

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter > threshold * min(area1, area2)


def _avg_conf(items):
    return 0.0 if not items else sum(i["conf"] for i in items) / len(items)


def _rotate_bound_gray(img, angle_deg):
    (h, w) = img.shape[:2]
    c_x, c_y = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((c_x, c_y), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    M[0, 2] += (n_w / 2) - c_x
    M[1, 2] += (n_h / 2) - c_y
    return cv2.warpAffine(
        img,
        M,
        (n_w, n_h),
        flags=cv2.INTER_CUBIC,
        borderValue=255,
    )


def _unsharp(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def _sauvola(gray, window=31, k=0.34, R=128):
    """Lightweight Sauvola threshold."""
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (window, window), normalize=True)
    sqmean = cv2.boxFilter(gray_f * gray_f, -1, (window, window),
                           normalize=True)
    var = np.maximum(sqmean - mean * mean, 0)
    std = np.sqrt(var)
    t = mean * (1 + k * ((std / (R + 1e-6)) - 1))
    return (gray_f > t).astype(np.uint8) * 255


def _ocr_try_configs(img_gray, bias_digits=True, min_conf=80):
    """Try a few Tesseract configs and keep the best text."""
    proc = (cv2.bitwise_not(img_gray)
            if np.mean(img_gray) < 128 else img_gray.copy())
    pil = Image.fromarray(proc)

    configs = [
        "--oem 1 --psm 7",
        "--oem 1 --psm 8",
        "--oem 1 --psm 6",
    ]
    if bias_digits:
        configs = [
            "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
            "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789",
        ] + configs

    best_text, best_conf = "", 0
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(
                pil,
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue

        tokens = []
        for i in range(len(data["text"])):
            t = (data["text"][i] or "").strip()
            if not t:
                continue
            try:
                c = int(float(data["conf"][i]))
            except Exception:
                c = -1
            if c > min_conf:
                tokens.append((t, c))

        if not tokens:
            continue

        text = " ".join(t for t, _ in tokens).strip()
        conf = int(round(sum(c for _, c in tokens) / len(tokens)))

        if bias_digits and text and not any(ch.isalpha() for ch in text):
            conf += 5

        if (conf > best_conf or
                (conf == best_conf and len(text) > len(best_text))):
            best_text, best_conf = text, conf

    return best_text, best_conf


def _extract_text_blob(roi_masked):
    """Focus on densest text in the ROI."""
    if np.mean(roi_masked) < 128:
        g = cv2.bitwise_not(roi_masked)
    else:
        g = roi_masked.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = _unsharp(g)

    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sau = _sauvola(g, window=25, k=0.32, R=128)
    bin_img = cv2.bitwise_and(otsu, sau)

    er = cv2.erode(bin_img, np.ones((2, 2), np.uint8), iterations=1)
    cl = cv2.morphologyEx(er, cv2.MORPH_CLOSE,
                          np.ones((3, 3), np.uint8), iterations=1)

    h, w = cl.shape
    border = np.zeros_like(cl)
    cv2.rectangle(border, (0, 0), (w - 1, h - 1), 255, 2)
    cl = cv2.subtract(cl, cv2.bitwise_and(cl, border))

    cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return g

    best_rect, best_score = None, -1
    for c in cnts:
        x, y, wc, hc = cv2.boundingRect(c)
        if wc < 6 or hc < 8:
            continue
        patch = cl[y:y + hc, x:x + wc]
        density = cv2.countNonZero(patch) / float(wc * hc)
        area = wc * hc
        score = area * (0.5 + density)
        if score > best_score:
            best_score = score
            best_rect = (x, y, wc, hc)

    if best_rect is None:
        return g

    x, y, wc, hc = best_rect
    pad = max(2, int(0.06 * max(wc, hc)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + wc + pad)
    y1 = min(h, y + hc + pad)
    return g[y0:y1, x0:x1]


# ---------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------
def _shape_name_from_vertices(nv):
    if nv == 3:
        return "triangle"
    if nv == 4:
        return "rectangle"
    if nv == 5:
        return "pentagon"
    if nv == 6:
        return "hexagon"
    if nv == 8:
        return "octagon"
    if nv > 8:
        return "circle"
    return f"polygon ({nv}-sided)"


def _run_tess_conf(patch_gray, cfg):
    """(kept for parity) – single tesseract call on a patch."""
    try:
        txt = pytesseract.image_to_string(
            patch_gray,
            config=cfg
        ).strip()
        if not txt:
            return "", 0

        d = pytesseract.image_to_data(
            patch_gray,
            config=cfg,
            output_type=pytesseract.Output.DICT,
        )
        confs = []
        for c in d.get("conf", []):
            try:
                v = int(float(c))
            except Exception:
                continue
            if v > MIN_CONF:
                confs.append(v)

        if not confs:
            return "", 0
        return txt, int(sum(confs) / len(confs))
    except Exception:
        return "", 0


# ---------------------------------------------------------------------
# shape-based OCR
# ---------------------------------------------------------------------
def find_text_in_shapes(cv_image_bgr):
    """Detect hollow symbols (circle/octagon/etc.) and OCR inside."""
    PASS2_LOG.info("PASS 2 (shape): starting …")

    gray_full = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_full, 240, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        PASS2_LOG.info("PASS 2 (shape): no contours found")
        return []

    hierarchy = hierarchy[0]
    total = len(contours)
    PASS2_LOG.info("PASS 2 (shape): candidate contours: %d", total)

    H, W = gray_full.shape
    results = []
    step = max(1, total // 20)
    analyzed = 0
    kept = 0

    for i, contour in enumerate(contours, start=1):
        has_child = (hierarchy[i - 1][2] != -1)
        if not has_child:
            if i == 1 or i % step == 0 or i == total:
                PASS2_LOG.info(
                    "PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)",
                    i, total, analyzed, kept,
                )
            continue

        area = cv2.contourArea(contour)
        if area < 800:
            if i == 1 or i % step == 0 or i == total:
                PASS2_LOG.info(
                    "PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)",
                    i, total, analyzed, kept,
                )
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        nv = len(approx)
        shape_name = _shape_name_from_vertices(nv)

        x, y, w, h = cv2.boundingRect(approx)
        if (w < 28 or h < 28 or
                w > 0.9 * W or h > 0.9 * H):
            if i == 1 or i % step == 0 or i == total:
                PASS2_LOG.info(
                    "PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)",
                    i, total, analyzed, kept,
                )
            continue

        roi_gray = gray_full[y:y + h, x:x + w]
        poly = (contour - [x, y]) if nv > 8 else (approx - [x, y])
        mask = np.zeros_like(roi_gray)
        cv2.drawContours(mask, [poly], -1, 255, -1)

        roi_masked = roi_gray.copy()
        roi_masked[mask == 0] = 255

        mean_val = np.mean(roi_masked[mask > 0])
        roi_proc = (cv2.bitwise_not(roi_masked)
                    if mean_val < 128 else roi_masked)

        analyzed += 1

        if (shape_name == "octagon" and
                analyzed <= 5 and
                SAVE_DEBUG_IMAGES):
            cv2.imwrite(f"debug_octagon_{analyzed}.png", roi_proc)

        text, conf = "", 0

        # digits first
        try:
            result = pytesseract.image_to_string(
                roi_proc,
                config=("--psm 8 "
                        "-c tessedit_char_whitelist=0123456789"),
            ).strip()
            if result:
                data = pytesseract.image_to_data(
                    roi_proc,
                    config=("--psm 8 "
                            "-c tessedit_char_whitelist=0123456789"),
                    output_type=pytesseract.Output.DICT,
                )
                confs = [
                    int(float(c))
                    for c in data["conf"]
                    if str(c).replace(".", "").replace("-", "").isdigit()
                    and int(float(c)) > SHAPE_MIN_CONF
                ]
                if confs:
                    text = result
                    conf = int(sum(confs) / len(confs))
        except Exception:
            pass

        # fallback: unrestricted
        if not text:
            try:
                result = pytesseract.image_to_string(
                    roi_proc,
                    config="--psm 8",
                ).strip()
                if result:
                    data = pytesseract.image_to_data(
                        roi_proc,
                        config="--psm 8",
                        output_type=pytesseract.Output.DICT,
                    )
                    confs = [
                        int(float(c))
                        for c in data["conf"]
                        if str(c).replace(".", "").replace("-", "").isdigit()
                        and int(float(c)) > SHAPE_MIN_CONF
                    ]
                    if confs:
                        text = result
                        conf = int(sum(confs) / len(confs))
            except Exception:
                pass

        results.append((text, (x, y, w, h), conf, shape_name))
        if text:
            kept += 1

        if i == 1 or i % step == 0 or i == total:
            PASS2_LOG.info(
                "PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)",
                i, total, analyzed, kept,
            )

    PASS2_LOG.info(
        "PASS 2 (shape): done. analyzed %d of %d; kept %d",
        analyzed, total, kept,
    )
    return results


# ---------------------------------------------------------------------
# main OCR pipeline
# ---------------------------------------------------------------------
def extract_text_and_create_debug_pdf(
    input_pdf: str,
    output_log: str,
    output_text_pdf: str,
) -> None:
    if not os.path.exists(input_pdf):
        print(f"Error: Input file not found at {input_pdf}")
        return

    print(f"Opening PDF: {input_pdf}")
    src_doc = fitz.open(input_pdf)
    num_pages = src_doc.page_count

    out_doc = fitz.open()
    all_text = []
    all_ocr_data = []

    print(f"Processing {num_pages} page(s)...")

    for page_num in range(num_pages):
        print(f"\nProcessing page {page_num + 1} / {num_pages}...")
        page = src_doc[page_num]

        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        cv_rgb = np.array(image)
        cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

        # line cleanup
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2,
        )
        binary_inv = cv2.bitwise_not(binary)
        min_len = 50
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
        h_mask = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN,
                                  h_kernel, iterations=1)
        v_mask = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN,
                                  v_kernel, iterations=1)
        lines_mask = cv2.add(h_mask, v_mask)
        lines_mask_d = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8),
                                  iterations=1)
        gray_clean = cv2.inpaint(gray, lines_mask_d, 3, cv2.INPAINT_TELEA)

        line_pixels = cv2.countNonZero(lines_mask)
        print(f"   > Removed lines (total pixels: {line_pixels}).")

        _, ocr_ready = cv2.threshold(
            gray_clean, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        img_pre = Image.fromarray(ocr_ready)

        # pass A – psm6
        print("   > Running PASS A: OCR PSM 6 ...")
        pass6 = _tess_image_to_data(img_pre, "--oem 1 --psm 6")
        pass6_data = [
            {
                "text": it["text"],
                "bbox": it["bbox"],
                "conf": it["conf"],
                "source": "psm6",
                "shape_type": None,
            }
            for it in pass6
        ]
        print(f"     - PASS A: {len(pass6_data)} words (>{MIN_CONF})")

        # pass B – psm12
        print("   > Running PASS B: OCR PSM 12 ...")
        pass12 = _tess_image_to_data(img_pre, "--oem 1 --psm 12")
        pass12_data = [
            {
                "text": it["text"],
                "bbox": it["bbox"],
                "conf": it["conf"],
                "source": "psm12",
                "shape_type": None,
            }
            for it in pass12
        ]
        print(f"     - PASS B: {len(pass12_data)} words (>{MIN_CONF})")

        # pass C – shape OCR
        print("   > Running PASS SHAPE ...")
        shape_results = find_text_in_shapes(cv_bgr)
        pass_shape_data = []
        for text, bbox, conf, shape_name in shape_results:
            x, y, w, h = bbox
            pass_shape_data.append(
                {
                    "text": text,
                    "bbox": (x, y, w, h),
                    "conf": conf,
                    "source": "shape_ocr",
                    "shape_type": shape_name,
                }
            )
        print(f"     - PASS SHAPE: {len(pass_shape_data)} items")

        OCR_LOG.info(
            "Page %d | psm6: %d (avg %.1f) | psm12: %d (avg %.1f) | shape: %d (avg %.1f)",
            page_num + 1,
            len(pass6_data), _avg_conf(pass6_data),
            len(pass12_data), _avg_conf(pass12_data),
            len(pass_shape_data), _avg_conf(pass_shape_data),
        )

        # merge
        print("   > Merging passes ...")
        word_data = list(pass6_data)

        # merge psm12
        for item2 in pass12_data:
            add = True
            replace_idx = -1
            for idx, item1 in enumerate(word_data):
                if _boxes_overlap(item1["bbox"], item2["bbox"]):
                    if (item2["conf"] > item1["conf"] or
                            (item2["conf"] == item1["conf"]
                             and len(item2["text"]) > len(item1["text"]))):
                        replace_idx = idx
                        print(
                            f"     - Replacing ({item1['source']}) "
                            f"'{item1['text']}' ({item1['conf']}) with "
                            f"({item2['source']}) '{item2['text']}' "
                            f"({item2['conf']})"
                        )
                    add = False
                    break
            if replace_idx >= 0:
                word_data[replace_idx] = item2
            elif add:
                word_data.append(item2)

        # merge shape
        for shape_item in pass_shape_data:
            matched = False
            for idx, existing in enumerate(word_data):
                if _boxes_overlap(existing["bbox"], shape_item["bbox"]):
                    if (shape_item["text"] and
                            shape_item["conf"] > existing["conf"]):
                        old_text = existing["text"]
                        old_conf = existing["conf"]
                        old_src = existing["source"]
                        word_data[idx] = shape_item
                        matched = True
                        print(
                            f"     - Replacing ({old_src}) '{old_text}' "
                            f"({old_conf}) with (shape_ocr) "
                            f"'{shape_item['text']}' ({shape_item['conf']})"
                        )
                    else:
                        existing["shape_type"] = shape_item["shape_type"]
                        matched = True
                        print(
                            f"     - Adding shape '{shape_item['shape_type']}' "
                            f"to ({existing['source']}) '{existing['text']}'"
                        )
                    break
            if not matched and shape_item["text"]:
                word_data.append(shape_item)
                print(
                    "     - Adding new shape_ocr result: "
                    f"'{shape_item['text']}' in {shape_item['shape_type']}"
                )

        contrib = {"psm6": 0, "psm12": 0, "shape_ocr": 0}
        for w in word_data:
            contrib[w["source"]] = contrib.get(w["source"], 0) + 1

        OCR_LOG.info(
            "Page %d merged: %d | psm6=%d psm12=%d shape=%d",
            page_num + 1,
            len(word_data),
            contrib.get("psm6", 0),
            contrib.get("psm12", 0),
            contrib.get("shape_ocr", 0),
        )
        print(f"     - Final merged results: {len(word_data)}")

        # copy original page to output
        try:
            out_doc.insert_pdf(src_doc,
                               from_page=page_num,
                               to_page=page_num)
        except TypeError:
            out_doc.insert_pdf(src_doc,
                               from_=page_num,
                               to=page_num)

        out_page = out_doc[-1]
        rect = page.rect
        x_scale = rect.width / pix.width
        y_scale = rect.height / pix.height

        # draw annotations
        for w in word_data:
            x, y, bw, bh = w["bbox"]
            r = fitz.Rect(
                x * x_scale,
                y * y_scale,
                (x + bw) * x_scale,
                (y + bh) * y_scale,
            )
            annot = out_page.add_highlight_annot(r)
            shape_label = w.get("shape_type") or "no shape"

            if w["source"] == "shape_ocr":
                annot.set_info(
                    content=w["text"],
                    title=f"Shape: {shape_label.capitalize()} "
                          f"(Conf: {w['conf']}%)",
                )
                annot.set_colors(stroke=(0, 1, 0))
            elif w["source"] == "psm12":
                annot.set_info(
                    content=w["text"],
                    title=f"PSM 12 (Shape: {shape_label}) "
                          f"(Conf: {w['conf']}%)",
                )
                annot.set_colors(stroke=(0, 0.6, 1))
            else:
                annot.set_info(
                    content=w["text"],
                    title=f"PSM 6 (Shape: {shape_label}) "
                          f"(Conf: {w['conf']}%)",
                )
                annot.set_colors(stroke=(1, 1, 0))

            annot.set_opacity(0.3)
            annot.update()

        # collect output
        all_ocr_data.append({"page": page_num + 1, "words": word_data})
        page_text = [w["text"] for w in word_data if w["text"]]
        all_text.append(
            f"=== Page {page_num + 1} ===\n" + "\n".join(page_text) + "\n"
        )
        print(f"Page {page_num + 1}: Recorded {len(page_text)} text elements")

    # write logs
    print(f"\nWriting text to {output_log} ...")
    with open(output_log, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    ocr_json_path = output_log.replace(".txt", "_ocr_data.json")
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(all_ocr_data, f, indent=2)
    print(f"Saved OCR data to {ocr_json_path}")

    print("\nCreating compact PDF ...")
    print(f"   - Saving OCR-highlight PDF: {output_text_pdf} ...")
    out_doc.save(
        output_text_pdf,
        garbage=4,
        deflate=True,
        clean=True,
    )

    src_doc.close()
    out_doc.close()

    print("\nPDF creation successful!")
    print("\nSummary:")
    print(f"   - Input: {input_pdf}")
    print(f"   - Text log: {output_log}")
    print(f"   - OCR/Text PDF: {output_text_pdf}")
    print(f"   - OCR Data JSON: {ocr_json_path}")
    print(f"   - Pages processed: {num_pages}")
    print("\nLegend:")
    print("   - Yellow = PSM 6")
    print("   - Blue   = PSM 12")
    print("   - Green  = Shape OCR")


if __name__ == "__main__":
    input_pdf = "plan.pdf"
    output_log = "logs.txt"
    output_text_pdf = "plan_text.pdf"

    try:
        extract_text_and_create_debug_pdf(
            input_pdf,
            output_log,
            output_text_pdf,
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
