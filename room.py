#!/usr/bin/env python3
"""
Label rooms/doors/windows from OCR + schedule and draw/emit walls from PDF.
Walls are detected from PDF vector data, merged, then snapped to horizontal/vertical.
"""
import json
import os
import sys
import math
import traceback

import fitz  # PyMuPDF
import colorsys

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
EXPECTED_SHAPES = {
    "Room": [],
    "Door": ["circle"],
    "Window": ["octagon"],
}

IMAGE_DPI = 300
MIN_FILLED_WALL_ASPECT_RATIO = 4
MIN_WALL_LENGTH_PDF = 1.5

GRAY_MIN_LIGHTNESS = 0.1
GRAY_MAX_LIGHTNESS = 0.9
RGB_TOLERANCE = 0.2
CMYK_COLOR_TOLERANCE = 0.1

WALL_LINE_COLOR_PDF = (1.0, 0.0, 0.0)
WALL_LINE_WIDTH_PDF = 0.8


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_json(path, fallback=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return fallback


def load_schedule(path):
    return load_json(path, fallback=None)


def load_ocr_data(path):
    data = load_json(path, fallback=None)
    if not data:
        return None
    valid = [item for item in data if isinstance(item, dict)]
    for item in valid:
        item["page"] = 1
    page_1 = [item for item in valid if item.get("page") == 1]
    if not page_1:
        page_1 = [{"page": 1, "words": []}]
    if "words" not in page_1[0]:
        page_1[0]["words"] = []
    return page_1


# ---------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------
def extract_room_marks(schedule):
    rooms = {}
    try:
        first = schedule.get("interiorFinishSchedule", {}).get("firstFloor", [])
        for room in first:
            if not isinstance(room, dict):
                continue
            num = str(room.get("number", "")).strip()
            name = str(room.get("NAME", "") or "").strip()
            if num:
                rooms[num] = name
    except Exception as e:
        print(f"Error extracting room marks: {e}")
    return rooms


def extract_door_window_marks(schedule):
    door_marks, window_marks = {}, {}
    try:
        for key, target in (("doorSchedule", door_marks), ("windowSchedule", window_marks)):
            for _, floor_data in schedule.get(key, {}).items():
                if not isinstance(floor_data, list):
                    continue
                for item in floor_data:
                    if not isinstance(item, dict):
                        continue
                    mark = str(item.get("MARK", "")).strip()
                    if mark:
                        target[mark] = item.get("TYPE", key[:-8].capitalize())
    except Exception as e:
        print(f"Error extracting door/window marks: {e}")
    return door_marks, window_marks


# ---------------------------------------------------------------------
# OCR lookups
# ---------------------------------------------------------------------
def find_rooms_in_ocr(ocr_data, room_marks):
    """Find room numbers and verify name components using spatial proximity."""
    found = {}
    if not ocr_data:
        return found
    words = ocr_data[0].get("words", [])
    if not words:
        return found

    SPATIAL_RADIUS = 800  # pixels - search within this radius for name components

    def get_bbox_center(bbox):
        """Calculate center coordinates of a bounding box [x, y, w, h]."""
        return (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)

    def euclidean_distance(center1, center2):
        """Calculate Euclidean distance between two points."""
        return math.hypot(center1[0] - center2[0], center1[1] - center2[1])

    for i, w in enumerate(words):
        if not isinstance(w, dict):
            continue
        text = str(w.get("text", "")).strip()
        bbox = w.get("bbox")
        if text not in room_marks:
            continue
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        if text in found:
            continue

        name = room_marks[text]
        wanted = {x for x in name.upper().split() if x}
        if not wanted:
            match = True
            ctx_txt = "N/A"
        else:
            # Use spatial proximity instead of word-count window
            room_center = get_bbox_center(bbox)
            ctx_set = set()
            ctx_list = []

            # Search all words on the page for those within spatial radius
            for other_word in words:
                if not isinstance(other_word, dict):
                    continue
                other_bbox = other_word.get("bbox")
                if not (isinstance(other_bbox, (list, tuple)) and len(other_bbox) == 4):
                    continue

                other_center = get_bbox_center(other_bbox)
                distance = euclidean_distance(room_center, other_center)

                if distance <= SPATIAL_RADIUS:
                    txt = str(other_word.get("text", "")).strip()
                    if txt:
                        ctx_set.add(txt.upper())
                        ctx_list.append(txt)

            match = wanted.issubset(ctx_set)
            ctx_txt = " ".join(ctx_list)

        if match:
            found[text] = {
                "name": name,
                "bbox": bbox,
                "page": ocr_data[0].get("page", 1),
                "context": ctx_txt,
                "shape": w.get("shape_type") or "N/A",
            }
    return found


def find_simple_marks_in_ocr(ocr_data, marks_dict, category):
    found = {}
    expected_shapes = EXPECTED_SHAPES.get(category, [])
    require_shape = bool(expected_shapes)
    if not ocr_data:
        return found
    words = ocr_data[0].get("words", [])
    if not words:
        return found

    for w in words:
        if not isinstance(w, dict):
            continue
        text = str(w.get("text", "")).strip()
        bbox = w.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue

        text_clean = text.strip("().,;:!?-")
        if text not in marks_dict and text_clean not in marks_dict:
            continue
        mark = text if text in marks_dict else text_clean
        shape = w.get("shape_type")
        if require_shape and shape not in expected_shapes:
            continue
        if mark in found:
            continue
        found[mark] = {
            "type": marks_dict[mark],
            "bbox": bbox,
            "page": ocr_data[0].get("page", 1),
            "shape": shape or "N/A",
        }
    return found


# ---------------------------------------------------------------------
# Wall detection + merge
# ---------------------------------------------------------------------
def merge_wall_segments(segments, gap_tolerance=25.0, parallel_tolerance=5.0):
    if not segments:
        return []

    def seg_len(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def colinear(seg1, seg2):
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3
        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 0.1 or len2 < 0.1:
            return False

        dot = abs((dx1 * dx2 + dy1 * dy2) / (len1 * len2))
        if dot < 0.95:
            return False

        def point_to_line(px, py):
            cross = abs((px - x1) * dy1 - (py - y1) * dx1)
            return cross / len1

        if point_to_line(x3, y3) > parallel_tolerance:
            return False
        if point_to_line(x4, y4) > parallel_tolerance:
            return False

        end1 = [(x1, y1), (x2, y2)]
        end2 = [(x3, y3), (x4, y4)]
        gap = min(seg_len(a, b) for a in end1 for b in end2)
        return gap <= gap_tolerance

    def merge_two(a, b):
        pts = [a[0], a[1], b[0], b[1]]
        best = (pts[0], pts[1])
        best_d = -1
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = seg_len(pts[i], pts[j])
                if d > best_d:
                    best_d = d
                    best = (pts[i], pts[j])
        return best

    merged = list(segments)
    for _ in range(10):
        changed = False
        out = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            cur = merged[i]
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                if colinear(cur, merged[j]):
                    cur = merge_two(cur, merged[j])
                    used.add(j)
                    changed = True
            out.append(cur)
            used.add(i)
        merged = out
        if not changed:
            break
    return merged


def axis_align_segment(seg):
    """Snap to horizontal or vertical (whichever is closer)."""
    (x1, y1), (x2, y2) = seg
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) >= abs(dy):  # horizontal
        return (x1, y1), (x2, y1)
    else:  # vertical
        return (x1, y1), (x1, y2)


def detect_walls_from_pdf(pdf_path, page_num=0, ocr_data=None):
    walls = []
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            print(f"Error: Page {page_num} not found.")
            return []
        page = doc[page_num]
        page_w, page_h = page.rect.width, page.rect.height

        # OCR exclusion zones (so text boxes don't become walls)
        text_excl = []
        if ocr_data:
            pdf_scale = 72.0 / IMAGE_DPI
            pad = 5 * pdf_scale
            for w in ocr_data[0].get("words", []):
                bbox = w.get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                x, y, bw, bh = bbox
                x0 = max(0, x * pdf_scale - pad)
                y0 = max(0, y * pdf_scale - pad)
                x1 = min(page_w, (x + bw) * pdf_scale + pad)
                y1 = min(page_h, (y + bh) * pdf_scale + pad)
                text_excl.append(fitz.Rect(x0, y0, x1, y1))

        def overlaps_text(r):
            return any(r.intersects(t) for t in text_excl)

        def is_gray(color):
            if not color:
                return False
            n = len(color)
            try:
                if n == 1:
                    g = color[0]
                    return GRAY_MIN_LIGHTNESS <= g <= GRAY_MAX_LIGHTNESS
                if n == 3:
                    r, g, b = color
                    light = (max(r, g, b) + min(r, g, b)) / 2
                    if not (GRAY_MIN_LIGHTNESS <= light <= GRAY_MAX_LIGHTNESS):
                        return False
                    return max(abs(r - g), abs(r - b), abs(g - b)) <= RGB_TOLERANCE
                if n == 4:
                    c, m, y, k = color
                    if c <= CMYK_COLOR_TOLERANCE and m <= CMYK_COLOR_TOLERANCE and y <= CMYK_COLOR_TOLERANCE:
                        light = 1.0 - k
                        return GRAY_MIN_LIGHTNESS <= light <= GRAY_MAX_LIGHTNESS
                    return False
            except Exception:
                return False
            return False

        for path in page.get_drawings():
            stroke = path.get("color")
            fill = path.get("fill")
            use_stroke = is_gray(stroke)
            use_fill = is_gray(fill)
            if not (use_stroke or use_fill):
                continue

            for item in path.get("items", []):
                kind = item[0]
                if kind == "l":
                    p1, p2 = item[1], item[2]
                    r = fitz.Rect(p1, p2)
                    if overlaps_text(r):
                        continue
                    length = math.hypot(p2.x - p1.x, p2.y - p1.y)
                    if length >= MIN_WALL_LENGTH_PDF and use_stroke:
                        walls.append(((p1.x, p1.y), (p2.x, p2.y)))
                elif kind == "re":
                    rect = item[1]
                    r = rect if isinstance(rect, fitz.Rect) else fitz.Rect(*rect)
                    if overlaps_text(r):
                        continue
                    w = abs(r.x1 - r.x0)
                    h = abs(r.y1 - r.y0)
                    if w == 0 or h == 0:
                        continue
                    ar = max(w, h) / min(w, h)
                    if ar >= MIN_FILLED_WALL_ASPECT_RATIO and use_fill:
                        if w > h:  # horizontal long
                            walls.append(((r.x0, r.y0), (r.x1, r.y0)))
                            walls.append(((r.x0, r.y1), (r.x1, r.y1)))
                        else:  # vertical long
                            walls.append(((r.x0, r.y0), (r.x0, r.y1)))
                            walls.append(((r.x1, r.y0), (r.x1, r.y1)))

        doc.close()
    except Exception as e:
        print(f"Error detecting walls: {e}")
        traceback.print_exc()
    return walls


# ---------------------------------------------------------------------
# Drawing / export
# ---------------------------------------------------------------------
def generate_distinct_colors(n):
    colors = []
    m = max(n, 10)
    for i in range(m):
        h = i / m
        l = 0.6 + (i % 3) * 0.1
        s = 0.8 + (i % 2) * 0.1
        try:
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            colors.append((max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))))
        except Exception:
            colors.append((1, 0, 0))
    return (colors[:n] if n <= m else colors * (n // m) + colors[: n % m]) or [(1, 0, 0)]


def create_annotated_pdf(input_pdf, rooms, doors, windows, walls, output_pdf):
    try:
        doc = fitz.open(input_pdf)
    except Exception as e:
        print(f"Error opening PDF '{input_pdf}': {e}")
        return

    page = doc[0]
    label_scale = 72.0 / IMAGE_DPI

    # walls
    shape = page.new_shape()
    for p1, p2 in walls:
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= 0.1:
            continue
        try:
            shape.draw_line(fitz.Point(*p1), fitz.Point(*p2))
        except Exception:
            pass
    shape.finish(color=WALL_LINE_COLOR_PDF, width=WALL_LINE_WIDTH_PDF)
    try:
        shape.commit()
    except Exception:
        pass

    door_color = (0, 0, 1)
    window_color = (0, 0.7, 0)
    text_color = (0, 0, 0)
    text_bg = (1, 1, 0.8)

    room_colors = generate_distinct_colors(len(rooms) or 1)
    room_color_map = {rn: room_colors[i % len(room_colors)] for i, rn in enumerate(sorted(rooms))}

    def label_box(bbox, text, stroke_color):
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return
        try:
            x, y, w, h = map(float, bbox)
        except Exception:
            return
        x0, y0 = x * label_scale, y * label_scale
        x1, y1 = x0 + w * label_scale, y0 + h * label_scale
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(page.rect.width, x1), min(page.rect.height, y1)

        try:
            hl = page.add_highlight_annot(fitz.Rect(x0, y0, x1, y1))
            hl.set_colors(stroke=stroke_color)
            hl.set_opacity(0.4)
            hl.update()
        except Exception:
            pass

        font_size = 8
        pad = 2
        try:
            text_w = fitz.get_text_length(text, fontsize=font_size)
        except Exception:
            text_w = len(text) * font_size * 0.6
        tx0, ty0 = x0, y0 - font_size - pad * 2
        tx1, ty1 = tx0 + text_w + pad * 2, ty0 + font_size + pad * 2
        if ty0 < 0:
            ty0 = y1 + pad
            ty1 = ty0 + font_size + pad * 2
        tx0 = max(pad, min(tx0, page.rect.width - text_w - pad * 2))
        tx1 = tx0 + text_w + pad * 2
        ty0 = max(pad, min(ty0, page.rect.height - font_size - pad * 2))
        ty1 = ty0 + font_size + pad * 2
        try:
            ann = page.add_freetext_annot(
                fitz.Rect(tx0, ty0, tx1, ty1),
                text,
                fontsize=font_size,
                text_color=text_color,
                fill_color=text_bg,
            )
            ann.set_opacity(0.85)
            ann.update()
        except Exception:
            pass

    for mark, info in doors.items():
        label_box(info.get("bbox"), f"DOOR {mark} ({info.get('shape', '?')})", door_color)

    for mark, info in windows.items():
        label_box(info.get("bbox"), f"WINDOW {mark} ({info.get('shape', '?')})", window_color)

    for rn, info in rooms.items():
        label_box(
            info.get("bbox"),
            f"ROOM {rn}: {info.get('name', '')}".strip(),
            room_color_map.get(rn, (1, 0, 0)),
        )

    # Count annotations added
    total_items = len(doors) + len(windows) + len(rooms)

    try:
        doc.save(output_pdf, garbage=4, deflate=True, clean=True)
        print(f"  - Added {total_items} item labels/highlights")
        print(f"  - Added {len(walls)} wall segments")
    except Exception as e:
        print(f"  ERROR: Could not save PDF: {e}")
    finally:
        try:
            doc.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    try:
        base = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        base = os.getcwd()

    schedule_path = os.path.join(base, "schedule.json")
    ocr_path = os.path.join(base, "logs_ocr_data.json")
    input_pdf = os.path.join(base, "plan.pdf")
    output_pdf = os.path.join(base, "plan_rooms.pdf")
    export_path = os.path.join(base, "room_map.json")

    print("Loading...")
    schedule = load_schedule(schedule_path)
    if not schedule:
        print("No schedule.")
        return
    ocr_data = load_ocr_data(ocr_path)
    if not ocr_data:
        print("No OCR.")
        return

    print("\nExtracting marks from schedule...")
    room_marks = extract_room_marks(schedule)
    door_marks, window_marks = extract_door_window_marks(schedule)
    print(f"  - Room marks in schedule: {len(room_marks)}")
    print(f"  - Door marks in schedule: {len(door_marks)}")
    print(f"  - Window marks in schedule: {len(window_marks)}")

    print("\nSearching for rooms, doors, and windows in OCR data...")
    rooms = find_rooms_in_ocr(ocr_data, room_marks)
    print(f"  - Rooms detected: {len(rooms)}")
    for rnum, rinfo in sorted(rooms.items()):
        print(f"    • Room {rnum}: {rinfo.get('name', 'N/A')}")

    doors = find_simple_marks_in_ocr(ocr_data, door_marks, "Door")
    print(f"  - Doors detected: {len(doors)}")
    if doors:
        print(f"    • Door marks: {', '.join(sorted(doors.keys()))}")

    windows = find_simple_marks_in_ocr(ocr_data, window_marks, "Window")
    print(f"  - Windows detected: {len(windows)}")
    if windows:
        print(f"    • Window marks: {', '.join(sorted(windows.keys()))}")

    print("\nDetecting and processing walls from PDF...")
    if os.path.exists(input_pdf):
        raw_walls = detect_walls_from_pdf(input_pdf, page_num=0, ocr_data=ocr_data)
        print(f"  - Raw wall segments detected: {len(raw_walls)}")
        merged = merge_wall_segments(raw_walls, gap_tolerance=25.0, parallel_tolerance=5.0)
        axis_walls = [axis_align_segment(seg) for seg in merged]
        reduction = len(raw_walls) - len(merged)
        reduction_pct = (reduction / len(raw_walls) * 100) if raw_walls else 0
        print(f"  - After merging: {len(merged)} segments (reduced {reduction} segments, {reduction_pct:.1f}%)")
        print(f"  - After axis alignment: {len(axis_walls)} wall segments")
    else:
        print(f"  ERROR: PDF '{input_pdf}' not found.")
        axis_walls = []

    export_data = {"rooms": {}, "doors": {}, "windows": {}, "walls": []}

    for rn, info in rooms.items():
        export_data["rooms"][rn] = {
            "number": rn,
            "name": info.get("name", "Unknown"),
            "bbox": info["bbox"],
            "shape": info.get("shape", "N/A"),
        }
    for dm, info in doors.items():
        export_data["doors"][dm] = {
            "mark": dm,
            "type": info.get("type", "Unknown"),
            "bbox": info["bbox"],
            "shape": "door",
        }
    for wm, info in windows.items():
        export_data["windows"][wm] = {
            "mark": wm,
            "type": info.get("type", "Unknown"),
            "bbox": info["bbox"],
            "shape": "window",
        }

    for p1, p2 in axis_walls:
        export_data["walls"].append(
            {
                "start": [round(p1[0], 1), round(p1[1], 1)],
                "end": [round(p2[0], 1), round(p2[1], 1)],
            }
        )

    print("\nExporting data to JSON...")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    # Count JSON lines
    with open(export_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    print(f"  Exported to: {export_path}")
    print(f"  - Rooms: {len(export_data['rooms'])}")
    print(f"  - Doors: {len(export_data['doors'])}")
    print(f"  - Windows: {len(export_data['windows'])}")
    print(f"  - Walls: {len(export_data['walls'])}")
    print(f"  - Total JSON lines: {line_count}")

    print("\nGenerating annotated PDF...")
    if rooms or doors or windows or axis_walls:
        create_annotated_pdf(input_pdf, rooms, doors, windows, axis_walls, output_pdf)
        print(f"  PDF saved to: {output_pdf}")
    else:
        print("  WARNING: Nothing to annotate - no items detected.")


if __name__ == "__main__":
    main()
