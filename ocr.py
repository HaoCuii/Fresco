import os
import cv2
import json
import shutil
import logging
import numpy as np
import pytesseract

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
PASS2_LOG = logging.getLogger("ocr.shape")

# ---------------- Tesseract discovery ----------------
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path
    else:
        print("Tesseract OCR not found. Please install it and ensure it's on PATH.")
        raise SystemExit(1)

MIN_CONF = 10

# ---------------- Helpers ----------------
# (No complex preprocessing needed - keep it simple!)


# ---------------- Shape-only OCR ----------------

def find_text_in_shapes_only(cv_image_bgr):
    """Detect geometric shapes and OCR only inside those shapes.
    Returns list of dicts with keys: text, conf, bbox(x,y,w,h), shape_type.
    """
    gray_full = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)

    # For architectural drawings with HOLLOW shapes (circles, octagons)
    # Use simple thresholding to detect dark lines/borders
    _, thresh = cv2.threshold(gray_full, 240, 255, cv2.THRESH_BINARY_INV)

    # Use RETR_TREE to get hierarchy (shapes with children = shapes with text inside)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        PASS2_LOG.info("No contours found")
        return []

    hierarchy = hierarchy[0]  # Remove extra dimension
    PASS2_LOG.info("Shapes detected: %d", len(contours))

    H, W = gray_full.shape
    results = []

    for i, contour in enumerate(contours):
        # Filter by hierarchy: only process contours WITH children (text inside)
        h = hierarchy[i]
        has_child = h[2] != -1
        if not has_child:
            continue
        area = cv2.contourArea(contour)
        if area < 800:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        nv = len(approx)
        if   nv == 3: shape_name = "triangle"
        elif nv == 4: shape_name = "rectangle"
        elif nv == 5: shape_name = "pentagon"
        elif nv == 6: shape_name = "hexagon"
        elif nv == 8: shape_name = "octagon"
        elif nv > 8:  shape_name = "circle"
        else:         shape_name = f"polygon ({nv}-sided)"

        x, y, w, h = cv2.boundingRect(approx)
        if w < 28 or h < 28: 
            continue
        if w > 0.9 * W or h > 0.9 * H:
            continue

        # Mask to INSIDE of shape only (don't OCR the border!)
        roi_gray = gray_full[y:y+h, x:x+w]
        poly = (contour - [x, y]) if nv > 8 else (approx - [x, y])
        mask = np.zeros_like(roi_gray)
        cv2.drawContours(mask, [poly], -1, 255, -1)

        # Apply mask - set outside to white
        roi_masked = roi_gray.copy()
        roi_masked[mask == 0] = 255

        # Invert to white background if dark
        mean_val = np.mean(roi_masked[mask > 0])  # Only check inside the mask
        if mean_val < 128:
            roi_proc = cv2.bitwise_not(roi_masked)
        else:
            roi_proc = roi_masked

        # Try simple OCR with PSM 8 (single word) - best for numbers
        text, conf = "", 0
        try:
            result = pytesseract.image_to_string(roi_proc, config='--psm 8').strip()
            if result:
                # Get confidence
                data = pytesseract.image_to_data(roi_proc, config='--psm 8', output_type=pytesseract.Output.DICT)
                confs = [int(float(c)) for c in data['conf'] if str(c).replace('.','').replace('-','').isdigit() and int(float(c)) > MIN_CONF]
                if confs:
                    text = result
                    conf = int(sum(confs) / len(confs))
        except:
            pass

        if text:  # ONLY keep shapes that yield text
            results.append({
                'text': text,
                'conf': int(conf),
                'bbox': (x, y, w, h),
                'shape_type': shape_name,
            })

    return results


# ---------------- Public API ----------------

def annotate_shapes(input_path: str = "test.png", output_path: str = "text_annotated.png", results_json: str | None = None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to load image: {input_path}")

    # Run shape-only OCR
    items = find_text_in_shapes_only(bgr)
    logging.info(f"Kept {len(items)} shapes with text")

    # Draw only the shapes that produced text
    annotated = bgr.copy()
    for it in items:
        x, y, w, h = it['bbox']
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 200, 0), 2)
        label = f"{it['text']}  ·  {it['shape_type']}  ·  {it['conf']}%"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y - 6 if y - 6 > th + 2 else y + h + th + 2
        cv2.rectangle(annotated, (x, ty - th - 4), (x + tw + 4, ty + 2), (0,0,0), -1)
        cv2.putText(annotated, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, annotated)
    logging.info(f"Saved annotated image to: {output_path}")

    if results_json:
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump({'shapes': items}, f, indent=2)
        logging.info(f"Saved results JSON to: {results_json}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Shape-only OCR: detect shapes and extract text inside them.")
    p.add_argument("input", nargs="?", default="test.png", help="Input image (default: test.png)")
    p.add_argument("output", nargs="?", default="text_annotated.png", help="Output annotated image (default: text_annotated.png)")
    p.add_argument("--json", dest="json_out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    try:
        annotate_shapes(args.input, args.output, results_json=args.json_out)
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
