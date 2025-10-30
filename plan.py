import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import cv2
import numpy as np
import json
import shutil
import logging
import math

# ---------------- Logging (terminal only) ----------------
LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(message)s"
)
PASS2_LOG = logging.getLogger("ocr.pass2")   # shape-based progress
OCR_LOG   = logging.getLogger("ocr.compare") # page-level comparisons & summaries

# ---------------- Configuration ----------------
# Find the Tesseract binary in a cross-platform way
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Common Windows install path fallback
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path
    else:
        print("Tesseract OCR not found. Please install it and ensure it's on PATH.")
        raise SystemExit(1)

DPI = 300  # Rendering DPI for OCR only (not embedded)
MIN_CONF = 10
SAVE_DEBUG_IMAGES = False  # set True to save preprocessing debug PNGs

# ---------------- Small helpers ----------------
def _tess_image_to_data(img, config):
    """Runs Tesseract and returns structured items with text/conf/bbox/source."""
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    items = []
    for i in range(len(data['text'])):
        text = (data['text'][i] or "").strip()
        try:
            conf = int(float(data['conf'][i]))
        except Exception:
            conf = -1
        if text and conf > MIN_CONF:
            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            items.append({'text': text, 'bbox': (x, y, w, h), 'conf': conf})
    return items

def _boxes_overlap(bbox1, bbox2, threshold=0.5):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return False
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    return intersection > threshold * min(area1, area2)

def _avg_conf(items):
    if not items:
        return 0.0
    return sum(it['conf'] for it in items) / len(items)

def _rotate_bound_gray(img, angle_deg):
    (h, w) = img.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderValue=255)

def _unsharp(gray):
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharp

def _sauvola(gray, window=31, k=0.34, R=128):
    # fast approximate Sauvola (no skimage needed)
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window,window), normalize=True)
    sqmean = cv2.boxFilter(gray_f*gray_f, ddepth=-1, ksize=(window,window), normalize=True)
    var = np.maximum(sqmean - mean*mean, 0)
    std = np.sqrt(var)
    T = mean * (1 + k * ((std / (R + 1e-6)) - 1))
    out = (gray_f > T).astype(np.uint8) * 255
    return out

def _ocr_try_configs(img_gray, bias_digits=True, min_conf=10):
    """
    Run several Tesseract configs and return best (text, conf).
    (Used for full-page passes; not used in shape OCR anymore.)
    """
    proc = cv2.bitwise_not(img_gray) if np.mean(img_gray) < 128 else img_gray.copy()
    pil = Image.fromarray(proc)
    configs = [
        "--oem 1 --psm 7",  # single line
        "--oem 1 --psm 8",  # single word
        "--oem 1 --psm 6",  # block
    ]
    if bias_digits:
        configs = [
            "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
            "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789",
        ] + configs

    best_text, best_conf = "", 0
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(pil, config=cfg, output_type=pytesseract.Output.DICT)
            tokens = []
            for i in range(len(data["text"])):
                t = (data["text"][i] or "").strip()
                if not t: continue
                c = data["conf"][i]
                try: c = int(float(c))
                except: c = -1
                if c > min_conf:
                    tokens.append((t, c))
            if tokens:
                text = " ".join([t for t,_ in tokens]).strip()
                conf = int(round(sum(c for _,c in tokens) / len(tokens)))
                if bias_digits and text and not any(ch.isalpha() for ch in text):
                    conf += 5
                if conf > best_conf or (conf == best_conf and len(text) > len(best_text)):
                    best_text, best_conf = text, conf
        except Exception:
            continue
    return best_text, best_conf

def _extract_text_blob(roi_masked):
    """
    MSER-like morphology to isolate text blob(s) within a masked ROI.
    Returns a cropped grayscale image focusing on the densest text region.
    """
    if np.mean(roi_masked) < 128:
        g = cv2.bitwise_not(roi_masked)
    else:
        g = roi_masked.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = _unsharp(g)

    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sau = _sauvola(g, window=25, k=0.32, R=128)
    bin_img = cv2.bitwise_and(otsu, sau)

    er = cv2.erode(bin_img, np.ones((2,2), np.uint8), iterations=1)
    cl = cv2.morphologyEx(er, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    h, w = cl.shape
    border = np.zeros_like(cl)
    cv2.rectangle(border, (0,0), (w-1, h-1), 255, 2)
    border_overlap = cv2.bitwise_and(cl, border)
    cl = cv2.subtract(cl, border_overlap)

    cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return g

    best_rect = None
    best_score = -1
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        if wc < 6 or hc < 8:
            continue
        patch = cl[y:y+hc, x:x+wc]
        density = cv2.countNonZero(patch) / float(wc*hc)
        area = wc * hc
        score = area * (0.5 + density)
        if score > best_score:
            best_score = score
            best_rect = (x,y,wc,hc)

    if best_rect is None:
        return g

    x,y,wc,hc = best_rect
    pad = max(2, int(0.06 * max(wc, hc)))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + wc + pad); y1 = min(h, y + hc + pad)
    return g[y0:y1, x0:x1]

# ---------------- Minimal helpers for shape OCR ----------------
def _shape_name_from_vertices(nv):
    if   nv == 3: return "triangle"
    elif nv == 4: return "rectangle"
    elif nv == 5: return "pentagon"
    elif nv == 6: return "hexagon"
    elif nv == 8: return "octagon"
    elif nv > 8:  return "circle"
    else:         return f"polygon ({nv}-sided)"

def _run_tess_conf(patch_gray, cfg):
    """Run a single Tesseract config; return (text, conf)."""
    try:
        txt = pytesseract.image_to_string(patch_gray, config=cfg).strip()
        if not txt:
            return "", 0
        d = pytesseract.image_to_data(patch_gray, config=cfg, output_type=pytesseract.Output.DICT)
        confs = []
        for c in d.get("conf", []):
            try:
                v = int(float(c))
                if v > MIN_CONF:
                    confs.append(v)
            except Exception:
                pass
        if not confs:
            return "", 0
        return txt, int(sum(confs) / len(confs))
    except Exception:
        return "", 0

# ---------------- Shape-based OCR (PSM 12 + PSM 7 only; no scales/angles) ----------------
def find_text_in_shapes(cv_image_bgr):
    """
    Shape OCR tuned for architectural HOLLOW shapes (circles, octagons, etc.).
    SPEED: Only two OCR configs per shape (PSM 12, then PSM 7). No angles/scales.
    Logs are quiet: periodic progress (analyzed/kept) + final summary.
    Returns: list of (text, (x,y,w,h), conf, shape_name)
    """
    PASS2_LOG.info("PASS 2 (shape): starting …")
    gray_full = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect dark borders on a light plan (works well for hollow symbols)
    _, thresh = cv2.threshold(gray_full, 240, 255, cv2.THRESH_BINARY_INV)

    # Use hierarchy to target shapes that CONTAIN something (child contour)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        PASS2_LOG.info("PASS 2 (shape): no contours found")
        return []
    hierarchy = hierarchy[0]

    total = len(contours)
    PASS2_LOG.info("PASS 2 (shape): candidate contours: %d", total)

    H, W = gray_full.shape
    results = []
    step = max(1, total // 20)  # ~5% steps

    analyzed = 0
    kept = 0

    for i, contour in enumerate(contours, start=1):
        has_child = (hierarchy[i-1][2] != -1)
        if not has_child:
            if i == 1 or i % step == 0 or i == total:
                PASS2_LOG.info("PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)", i, total, analyzed, kept)
            continue

        area = cv2.contourArea(contour)
        if area < 800:
            if i == 1 or i % step == 0 or i == total:
                PASS2_LOG.info("PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)", i, total, analyzed, kept)
            continue

        peri = cv2.arcLength(contour, True)
        # Use tighter approximation (0.01) to better distinguish octagons from circles
        # Lower epsilon = more vertices preserved = better shape detection
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        nv = len(approx)
        shape_name = _shape_name_from_vertices(nv)

        x, y, w, h = cv2.boundingRect(approx)
        if w < 28 or h < 28 or w > 0.9 * W or h > 0.9 * H:
            if i == 1 or i % step == 0 or i == total:
                PASS2_LOG.info("PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)", i, total, analyzed, kept)
            continue

        # EXACT LOGIC FROM OCR.PY - mask inside only
        roi_gray = gray_full[y:y+h, x:x+w]
        poly = (contour - [x, y]) if nv > 8 else (approx - [x, y])
        mask = np.zeros_like(roi_gray)
        cv2.drawContours(mask, [poly], -1, 255, -1)

        roi_masked = roi_gray.copy()
        roi_masked[mask == 0] = 255

        # Invert to white background if dark
        mean_val = np.mean(roi_masked[mask > 0])
        if mean_val < 128:
            roi_proc = cv2.bitwise_not(roi_masked)
        else:
            roi_proc = roi_masked

        analyzed += 1

        # DEBUG: Save first 5 octagons for inspection
        if shape_name == "octagon" and analyzed <= 5 and SAVE_DEBUG_IMAGES:
            cv2.imwrite(f"debug_octagon_{analyzed}_roi_proc.png", roi_proc)
            PASS2_LOG.info(f"DEBUG: Saved debug_octagon_{analyzed}_roi_proc.png")

        # OCR with digit whitelist for better number recognition (prevents '203' -> 'G03')
        text, conf = "", 0

        # Try digits-only first (best for room numbers)
        try:
            result = pytesseract.image_to_string(roi_proc, config='--psm 8 -c tessedit_char_whitelist=0123456789').strip()
            if result:
                data = pytesseract.image_to_data(roi_proc, config='--psm 8 -c tessedit_char_whitelist=0123456789', output_type=pytesseract.Output.DICT)
                confs = [int(float(c)) for c in data['conf'] if str(c).replace('.','').replace('-','').isdigit() and int(float(c)) > MIN_CONF]
                if confs:
                    text = result
                    conf = int(sum(confs) / len(confs))
                    if shape_name == "octagon" and SAVE_DEBUG_IMAGES:
                        PASS2_LOG.info(f"DEBUG: Octagon #{analyzed} -> '{text}' (conf: {conf}) [DIGITS-ONLY]")
        except:
            pass

        # Fallback: try unrestricted if digits-only failed
        if not text:
            try:
                result = pytesseract.image_to_string(roi_proc, config='--psm 8').strip()
                if result:
                    data = pytesseract.image_to_data(roi_proc, config='--psm 8', output_type=pytesseract.Output.DICT)
                    confs = [int(float(c)) for c in data['conf'] if str(c).replace('.','').replace('-','').isdigit() and int(float(c)) > MIN_CONF]
                    if confs:
                        text = result
                        conf = int(sum(confs) / len(confs))
                        if shape_name == "octagon" and SAVE_DEBUG_IMAGES:
                            PASS2_LOG.info(f"DEBUG: Octagon #{analyzed} -> '{text}' (conf: {conf}) [UNRESTRICTED]")
            except:
                pass

        # ALWAYS add shape (even without text) for later shape_type matching
        results.append((text, (x, y, w, h), conf, shape_name))
        if text:
            kept += 1

        if i == 1 or i % step == 0 or i == total:
            PASS2_LOG.info("PASS 2 (shape): progress %d/%d (analyzed=%d, kept=%d)", i, total, analyzed, kept)

    PASS2_LOG.info("PASS 2 (shape): done. analyzed %d of %d; kept %d", analyzed, total, kept)
    return results

# ---------------- Core ----------------
def extract_text_and_create_debug_pdf(input_pdf, output_log, output_text_pdf):
    """
    OCR a PDF by rendering each page to an image for Tesseract,
    but keep the output PDF lightweight by:
      - copying the original vector page into the output, and
      - overlaying highlight annotations (no raster background).
    """
    if not os.path.exists(input_pdf):
        print(f"Error: Input file not found at {input_pdf}")
        return

    print(f"Opening PDF: {input_pdf}")
    src_doc = fitz.open(input_pdf)
    num_pages = src_doc.page_count

    all_text = []
    all_ocr_data = []

    # Output PDF (will receive copied pages + highlight annots)
    out_doc = fitz.open()

    print(f"Processing {num_pages} page(s)...")

    for page_num in range(num_pages):
        print(f"\nProcessing page {page_num + 1} / {num_pages}...")
        page = src_doc[page_num]

        # ---------- Render page to image (for OCR only) ----------
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False for smaller temp images

        # Convert Pixmap -> PIL Image
        img_data = pix.tobytes("png")  # memory only; not embedding this
        image = Image.open(io.BytesIO(img_data))

        # ---------- Pre-process for OCR ----------
        cv_image_rgb = np.array(image)
        cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold (robust across backgrounds)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        binary_inv = cv2.bitwise_not(binary)

        # Line detection & removal
        MIN_LINE_LENGTH = 50
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MIN_LINE_LENGTH, 1))
        detect_horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, MIN_LINE_LENGTH))
        detect_vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        lines_mask = cv2.add(detect_horizontal, detect_vertical)
        dilate_kernel = np.ones((3, 3), np.uint8)
        lines_mask_dilated = cv2.dilate(lines_mask, dilate_kernel, iterations=1)
        gray_no_lines = cv2.inpaint(gray, lines_mask_dilated, 3, cv2.INPAINT_TELEA)

        line_pixels = cv2.countNonZero(lines_mask)
        print(f"   > Removed lines (total pixels: {line_pixels}).")

        # Final binarization for Tesseract
        _, ocr_ready_image = cv2.threshold(
            gray_no_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        image_preprocessed = Image.fromarray(ocr_ready_image)

        # Optional debug images
        if SAVE_DEBUG_IMAGES:
            debug_basename = os.path.splitext(output_text_pdf)[0]
            image1_filename = f"{debug_basename}_text_only_page_{page_num + 1}.png"
            try:
                image_preprocessed.save(image1_filename, "PNG")
                print(f"   > Saved text-only image to: {image1_filename}")
            except Exception as e:
                print(f"   > Warning: Could not save text-only image. Error: {e}")

            lines_only_image = cv2.bitwise_not(lines_mask)
            image2_preprocessed = Image.fromarray(lines_only_image)
            image2_filename = f"{debug_basename}_lines_only_page_{page_num + 1}.png"
            try:
                image2_preprocessed.save(image2_filename, "PNG")
                print(f"   > Saved lines-only image (wall map) to: {image2_filename}")
            except Exception as e:
                print(f"   > Warning: Could not save lines-only image. Error: {e}")

        # ---------- PASS A: PSM 6 ----------
        print(f"   > Running PASS A: OCR PSM 6 on page {page_num + 1}...")
        config_psm6 = r'--oem 1 --psm 6'
        pass6_items = _tess_image_to_data(image_preprocessed, config_psm6)
        pass6_data = [{'text': it['text'], 'bbox': it['bbox'], 'conf': it['conf'], 'source': 'psm6', 'shape_type': None}
                        for it in pass6_items]
        print(f"     - PASS A: Found {len(pass6_data)} words (conf > {MIN_CONF})")

        # ---------- PASS B: PSM 12 ----------
        print(f"   > Running PASS B: OCR PSM 12 on page {page_num + 1}...")
        config_psm12 = r'--oem 1 --psm 12'
        pass12_items = _tess_image_to_data(image_preprocessed, config_psm12)
        pass12_data = [{'text': it['text'], 'bbox': it['bbox'], 'conf': it['conf'], 'source': 'psm12', 'shape_type': None}
                          for it in pass12_items]
        print(f"     - PASS B: Found {len(pass12_data)} words (conf > {MIN_CONF})")

        # ---------- PASS SHAPE: Shape-based OCR (PSM12 + PSM7 only) ----------
        print(f"   > Running PASS SHAPE: Shape-based OCR on page {page_num + 1}...")
        shape_results = find_text_in_shapes(cv_image_bgr)  # (text_full, bbox, conf_avg, shape_name)
        pass_shape_data = []
        for text, bbox, conf, shape_name in shape_results:
            x, y, w, h = bbox
            pass_shape_data.append({
                'text': text,
                'bbox': (x, y, w, h),
                'conf': conf,
                'source': 'shape_ocr',
                'shape_type': shape_name
            })
        print(f"     - PASS SHAPE: Found {len(pass_shape_data)} text elements in shapes")

        # ---------- Page-level comparison logging ----------
        OCR_LOG.info(
            "Page %d comparison | psm6: %d (avg conf %.1f) | psm12: %d (avg conf %.1f) | shape: %d (avg conf %.1f)",
            page_num + 1,
            len(pass6_data), _avg_conf(pass6_data),
            len(pass12_data), _avg_conf(pass12_data),
            len(pass_shape_data), _avg_conf(pass_shape_data)
        )

        # ---------- Merge results: higher confidence wins when overlapping ----------
        print(f"   > Merging results from all passes...")
        word_data = []

        # Start with PSM 6 results
        for item in pass6_data:
            word_data.append(item)

        # Merge PSM 12
        for item2 in pass12_data:
            should_add = True
            replace_idx = -1
            for idx, item1 in enumerate(word_data):
                if _boxes_overlap(item1['bbox'], item2['bbox']):
                    if item2['conf'] > item1['conf'] or (item2['conf'] == item1['conf'] and len(item2['text']) > len(item1['text'])):
                        replace_idx = idx
                        print(f"     - Replacing ({item1['source']}) '{item1['text']}' ({item1['conf']}) "
                              f"with ({item2['source']}) '{item2['text']}' ({item2['conf']})")
                    should_add = False
                    break
            if replace_idx >= 0:
                word_data[replace_idx] = item2
            elif should_add:
                word_data.append(item2)

        # Merge SHAPE:
        # - If shape_ocr has higher confidence, REPLACE the PSM result
        # - Otherwise, just add shape_type to the existing PSM result
        # - If shape has text but didn't match any PSM, add it as a new entry
        for shape_item in pass_shape_data:
            matched_with_existing = False
            for idx, existing_item in enumerate(word_data):
                if _boxes_overlap(existing_item['bbox'], shape_item['bbox']):
                    # If shape_ocr has text AND higher confidence, replace the entire entry
                    if shape_item['text'] and shape_item['conf'] > existing_item['conf']:
                        old_text = existing_item['text']
                        old_conf = existing_item['conf']
                        old_source = existing_item['source']
                        word_data[idx] = shape_item
                        matched_with_existing = True
                        print(f"     - Replacing ({old_source}) '{old_text}' ({old_conf}) "
                              f"with (shape_ocr) '{shape_item['text']}' ({shape_item['conf']})")
                    else:
                        # Just add shape_type to existing result
                        word_data[idx]['shape_type'] = shape_item['shape_type']
                        matched_with_existing = True
                        print(f"     - Adding shape '{shape_item['shape_type']}' to ({existing_item['source']}) '{existing_item['text']}'")
                    break  # Only match with first overlapping item
            if not matched_with_existing and shape_item['text']:
                word_data.append(shape_item)
                print(f"     - Adding new shape_ocr result: '{shape_item['text']}' in {shape_item['shape_type']}")

        # Contribution breakdown
        contrib = {'psm6': 0, 'psm12': 0, 'shape_ocr': 0}
        for w in word_data:
            contrib[w['source']] = contrib.get(w['source'], 0) + 1
        OCR_LOG.info(
            "Page %d merged total: %d | contributions -> psm6: %d, psm12: %d, shape: %d",
            page_num + 1, len(word_data), contrib.get('psm6', 0), contrib.get('psm12', 0), contrib.get('shape_ocr', 0)
        )

        print(f"     - Final merged results: {len(word_data)} text elements")

        # ---------- Copy original page into OUT PDF (no raster background) ----------
        try:
            out_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
        except TypeError:
            out_doc.insert_pdf(src_doc, from_=page_num, to=page_num)

        out_page = out_doc[-1]

        # ---------- Place highlight annotations using scaled OCR boxes ----------
        rect = page.rect
        x_scale = rect.width / pix.width
        y_scale = rect.height / pix.height

        for w in word_data:
            x, y, bw, bh = w['bbox']
            pdf_rect = fitz.Rect(x * x_scale, y * y_scale, (x + bw) * x_scale, (y + bh) * y_scale)
            annot = out_page.add_highlight_annot(pdf_rect)

            shape_label = w.get('shape_type') or 'no shape'

            if w['source'] == 'shape_ocr':
                annot.set_info(content=w['text'], title=f"Shape: {shape_label.capitalize()} (Conf: {w['conf']}%)")
                annot.set_colors(stroke=(0, 1, 0))  # Green for shape-based OCR
            elif w['source'] == 'psm12':
                annot.set_info(content=w['text'], title=f"PSM 12 (Shape: {shape_label}) (Conf: {w['conf']}%)")
                annot.set_colors(stroke=(0, 0.6, 1))  # Blue-ish for PSM12
            else:
                annot.set_info(content=w['text'], title=f"PSM 6 (Shape: {shape_label}) (Conf: {w['conf']}%)")
                annot.set_colors(stroke=(1, 1, 0))  # Yellow for PSM6

            annot.set_opacity(0.3)
            annot.update()

        # ---------- Collect text/log data ----------
        all_ocr_data.append({'page': page_num + 1, 'words': word_data})
        page_text = [w['text'] for w in word_data if w['text']]
        all_text.append(f"=== Page {page_num + 1} ===\n" + "\n".join(page_text) + "\n")

        print(f"Page {page_num + 1}: Recorded {len(page_text)} text elements")

    # ---------- Persist logs ----------
    print(f"\nWriting text to {output_log}...")
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    ocr_json_path = output_log.replace('.txt', '_ocr_data.json')
    with open(ocr_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_ocr_data, f, indent=2)  # contains the 'shape' key
    print(f"Saved OCR data to {ocr_json_path}")

    # ---------- Save output PDF with cleanup/compression ----------
    print(f"\nCreating compact PDF...")
    print(f"   - Saving OCR-highlight PDF: {output_text_pdf}...")
    out_doc.save(
        output_text_pdf,
        garbage=4,     # deep garbage collection
        deflate=True,  # compress streams
        clean=True     # clean and rebuild xref
    )

    # Close documents
    src_doc.close()
    out_doc.close()

    print(f"\nPDF creation successful!")
    print(f"\nSummary:")
    print(f"   - Input: {input_pdf}")
    print(f"   - Text log: {output_log}")
    print(f"   - OCR/Text PDF: {output_text_pdf}")
    print(f"   - OCR Data JSON: {ocr_json_path}")
    print(f"   - Pages processed: {num_pages}")
    print(f"\nNOTE:")
    print(f"   - {output_text_pdf} (hover over highlights for info):")
    print(f"     - Yellow highlights = PSM 6 text (Shape: no shape)")
    print(f"     - Blue highlights   = PSM 12 text (Shape: no shape)")
    print(f"     - Green highlights  = Shape-based OCR (PSM12/PSM7)")
    print(f"   - {ocr_json_path} contains a 'shape' key for every word.")
    print(f"   - Vector page preserved (small file size)")

if __name__ == "__main__":
    # File paths
    input_pdf = "plan.pdf"
    output_log = "logs.txt"
    output_text_pdf = "plan_text.pdf"

    try:
        extract_text_and_create_debug_pdf(
            input_pdf,
            output_log,
            output_text_pdf
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    