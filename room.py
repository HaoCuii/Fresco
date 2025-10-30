#!/usr/bin/env python3
"""
Room labeling script that uses OCR results from plan.py
to identify room, door, and window locations, detects potential GRAY walls
(both filled rectangles and long lines) directly from PDF vector data 
(excluding text regions), and annotates the PDF marking these elements.
"""
import json
import fitz  # PyMuPDF
import os
import math
import colorsys  # For generating distinct colors
from itertools import combinations
import traceback  # For error printing

# --- Define Expected Shapes ---
EXPECTED_SHAPES = {
    "Room": [],  # No shape required for rooms
    "Door": ["circle"],
    "Window": ["octagon"]
}
# --- End Definitions ---

# --- Configuration for Wall Detection ---
# IMAGE_DPI is used for converting OCR bboxes to PDF coordinates
IMAGE_DPI = 300  # DPI used for OCR rendering

# General wall detection parameters
MIN_FILLED_WALL_ASPECT_RATIO = 4  # Minimum aspect ratio for elongated rectangles
MIN_WALL_LENGTH_PDF = 1.5  # Minimum length in PDF units for filled rects and lines

# Gray color detection thresholds
GRAY_MIN_LIGHTNESS = 0.1  # Min lightness (0.0=black, 1.0=white)
GRAY_MAX_LIGHTNESS = 0.9  # Max lightness
RGB_TOLERANCE = 0.2  # Max difference between R,G,B components
CMYK_COLOR_TOLERANCE = 0.1  # Max C,M,Y value for a CMYK gray

# Annotation parameters
WALL_LINE_COLOR_PDF = (1.0, 0.0, 0.0)  # Red color for detected walls
WALL_LINE_WIDTH_PDF = 0.8  # Thinner line width for drawing detected segments
# --- End Configuration ---


def load_schedule(schedule_path):
    """Load the schedule JSON file."""
    try:
        with open(schedule_path, 'r') as f: return json.load(f)
    except Exception as e: print(f"Error loading schedule '{schedule_path}': {e}"); return None


def load_ocr_data(ocr_data_path):
    """Load the OCR data JSON file from plan.py (filters for page 1)."""
    try:
        with open(ocr_data_path, 'r') as f: data = json.load(f)
        valid_data = [item for item in data if isinstance(item, dict)]
        for item in valid_data: item['page'] = 1  # Force page 1
        page_1_data = [item for item in valid_data if item.get('page') == 1]
        if not page_1_data: print("Warning: No OCR data found for page 1.")
        # Ensure 'words' key exists even if empty
        if page_1_data and 'words' not in page_1_data[0]:
            page_1_data[0]['words'] = []
        elif not page_1_data:  # Handle case where ocr file is empty or invalid
            page_1_data = [{'page': 1, 'words': []}]

        return page_1_data
    except Exception as e: print(f"Error loading OCR data '{ocr_data_path}': {e}"); return None


def extract_room_marks(schedule):
    """Extract first floor room marks."""
    room_marks = {}
    try:
        floor_data = schedule.get('interiorFinishSchedule', {}).get('firstFloor', [])
        if isinstance(floor_data, list):
            for room in floor_data:
                if isinstance(room, dict) and 'number' in room:
                    room_name = str(room.get('NAME', "") or "").strip()
                    room_number_str = str(room.get('number', '')).strip()
                    if room_number_str: room_marks[room_number_str] = room_name
    except Exception as e: print(f"Error extracting room marks: {e}")
    return room_marks


def extract_door_window_marks(schedule):
    """Extract all door and window marks."""
    door_marks = {}; window_marks = {}
    try:
        for schedule_key, marks_dict in [('doorSchedule', door_marks), ('windowSchedule', window_marks)]:
            for floor_name, floor_data in schedule.get(schedule_key, {}).items():
                if isinstance(floor_data, list):
                    for item in floor_data:
                        if isinstance(item, dict):
                            mark = str(item.get('MARK', '')).strip()
                            if mark: marks_dict[mark] = item.get('TYPE', schedule_key[:-8].capitalize())
    except Exception as e: print(f"Error extracting door/window marks: {e}")
    return door_marks, window_marks


def get_bbox_center(bbox):
    """Calculates center (x, y) from (x, y, w, h) bbox."""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4): return None
    try: x, y, w, h = map(float, bbox); assert w > 0 and h > 0; return (x + w / 2.0, y + h / 2.0)
    except: return None


def find_rooms_in_ocr(ocr_data, room_marks):
    """Search OCR for room numbers, verify name context."""
    # (Implementation remains the same)
    found_rooms = {}
    CONTEXT_WINDOW = 25
    if not ocr_data: return found_rooms
    page_item = ocr_data[0] if ocr_data else {}; page_num = page_item.get('page', 1)
    words_on_page = page_item.get('words', [])
    if not words_on_page: return found_rooms
    for i, word_item in enumerate(words_on_page):
        if not isinstance(word_item, dict): continue
        ocr_text = str(word_item.get('text', '')).strip(); bbox = word_item.get('bbox')
        is_valid_bbox = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox)
        if ocr_text in room_marks and is_valid_bbox:
            room_num = ocr_text; room_name = room_marks[room_num]; shape_type = word_item.get('shape_type'
            '')
            if room_num in found_rooms: continue
            name_components = set(w for w in room_name.upper().split() if w); is_match = False; context_text = ""
            if not name_components: is_match = True; context_text = "N/A"
            else:
                start_index = max(0, i - CONTEXT_WINDOW); end_index = min(len(words_on_page), i + CONTEXT_WINDOW + 1)
                context_words_set = set(); context_text_list = []
                for j in range(start_index, end_index):
                    if 0 <= j < len(words_on_page):
                        context_word_item = words_on_page[j]
                        if isinstance(context_word_item, dict) and context_word_item.get('text'):
                            word_text = str(context_word_item['text']).strip()
                            if word_text: context_words_set.add(word_text.upper()); context_text_list.append(word_text)
                is_match = name_components.issubset(context_words_set); context_text = " ".join(context_text_list)
            if is_match: found_rooms[room_num] = {'name': room_name, 'bbox': bbox, 'page': page_num, 'context': context_text, 'shape': shape_type or 'N/A'}
    return found_rooms


def find_simple_marks_in_ocr(ocr_data, marks_dict, item_category):
    """Find door/window marks matching shape rules."""
    # (Implementation remains the same)
    found_items = {}
    expected_shapes = EXPECTED_SHAPES.get(item_category, []); require_specific_shape = bool(expected_shapes)
    if not ocr_data: return found_items
    page_item = ocr_data[0] if ocr_data else {}; page_num = page_item.get('page', 1)
    words_on_page = page_item.get('words', [])
    if not words_on_page: return found_items
    for word_item in words_on_page:
        if not isinstance(word_item, dict): continue
        ocr_text = str(word_item.get('text', '')).strip(); bbox = word_item.get('bbox')
        is_valid_bbox = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox)
        # Strip parentheses and punctuation for matching: "(203)" or "205." matches schedule mark "203" or "205"
        ocr_text_clean = ocr_text.strip('().,;:!?-')
        if (ocr_text in marks_dict or ocr_text_clean in marks_dict) and is_valid_bbox:
            # Use the clean version to look up in marks_dict if original didn't match
            matched_mark = ocr_text if ocr_text in marks_dict else ocr_text_clean
            shape_type = word_item.get('shape_type')
            is_potential_match = False
            if require_specific_shape:
                # Accept any source (psm6, psm12, shape_ocr) as long as shape matches
                if shape_type in expected_shapes: is_potential_match = True
            else: is_potential_match = True
            if is_potential_match and matched_mark not in found_items:
                found_items[matched_mark] = {'type': marks_dict[matched_mark], 'bbox': bbox, 'page': page_num, 'shape': shape_type or 'N/A'}
    return found_items


# --- Detect GRAY Walls Directly from PDF Vector Data ---
def detect_walls_from_pdf(pdf_path, page_num=0, ocr_data=None):
    """
    Detects potential GRAY wall lines directly from PDF vector paths.
    Looks for gray-colored lines and rectangles, excluding text regions.
    Returns a list of wall segments [((x1, y1), (x2, y2)), ...] in PDF coordinates.
    """
    wall_segments_pdf = []
    
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            print(f"Error: Page {page_num} not found in PDF")
            return []

        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        print(f"     Processing PDF page {page_num + 1} for gray walls from vector data...")

        # Create text exclusion regions from OCR data
        text_exclusion_rects = []
        if ocr_data:
            page_item = ocr_data[0] if ocr_data else {}
            for word in page_item.get('words', []):
                bbox = word.get('bbox')
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        # Convert from IMAGE_DPI to PDF coordinates
                        pdf_scale = 72.0 / IMAGE_DPI
                        x, y, w_box, h_box = bbox
                        padding = 5 * pdf_scale  # Add padding in PDF units
                        x_pdf = x * pdf_scale
                        y_pdf = y * pdf_scale
                        w_pdf = w_box * pdf_scale
                        h_pdf = h_box * pdf_scale

                        rect = fitz.Rect(
                            max(0, x_pdf - padding),
                            max(0, y_pdf - padding),
                            min(page_width, x_pdf + w_pdf + padding),
                            min(page_height, y_pdf + h_pdf + padding)
                        )
                        text_exclusion_rects.append(rect)
                    except (ValueError, TypeError):
                        continue

        print(f"       Excluded {len(text_exclusion_rects)} OCR text regions.")

        def is_gray_color(color):
            """
            Check if a color tuple represents gray (RGB, Grayscale, or CMYK).
            Uses global config values: GRAY_MIN_LIGHTNESS, GRAY_MAX_LIGHTNESS, etc.
            """
            if not color:
                return False
                
            n_components = len(color)

            try:
                if n_components == 1:
                    # Grayscale (g) - 0.0=black, 1.0=white
                    g = color[0]
                    return GRAY_MIN_LIGHTNESS <= g <= GRAY_MAX_LIGHTNESS

                elif n_components == 3:
                    # RGB (r, g, b) - 0.0=black, 1.0=white
                    r, g, b = color
                    
                    # Check lightness range
                    lightness = (max(r, g, b) + min(r, g, b)) / 2
                    if not (GRAY_MIN_LIGHTNESS <= lightness <= GRAY_MAX_LIGHTNESS):
                        return False
                        
                    # Check if components are similar (not too colorful)
                    max_diff = max(abs(r - g), abs(r - b), abs(g - b))
                    return max_diff <= RGB_TOLERANCE

                elif n_components == 4:
                    # CMYK (c, m, y, k)
                    c, m, y, k = color
                    
                    if c <= CMYK_COLOR_TOLERANCE and m <= CMYK_COLOR_TOLERANCE and y <= CMYK_COLOR_TOLERANCE:
                        # 'k' (black) value. 0.0=white, 1.0=black.
                        # This is inverted from RGB/Grayscale lightness.
                        lightness = 1.0 - k
                        return GRAY_MIN_LIGHTNESS <= lightness <= GRAY_MAX_LIGHTNESS
                    return False
                    
            except Exception:
                return False  # Invalid color tuple

            return False

        def overlaps_text(rect):
            """Check if a rectangle overlaps with any text exclusion region."""
            for text_rect in text_exclusion_rects:
                if rect.intersects(text_rect):
                    return True
            return False

        # Extract drawing paths from the page
        paths = page.get_drawings()

        print(f"       Found {len(paths)} drawing paths in PDF.")

        gray_lines_found = 0
        gray_rects_found = 0

        for path in paths:
            stroke_color = path.get('color')
            fill_color = path.get('fill')

            is_gray_stroke = is_gray_color(stroke_color)
            is_gray_fill = is_gray_color(fill_color)

            if not (is_gray_stroke or is_gray_fill):
                continue

            items = path.get('items', [])

            for item_type, *points in items:
                if item_type == 'l':  # Line segment
                    if len(points) >= 2:
                        p1_fz, p2_fz = points[0], points[1]
                        
                        # Convert to simple tuples for easier handling
                        p1 = (p1_fz.x, p1_fz.y) if hasattr(p1_fz, 'x') else tuple(p1_fz)
                        p2 = (p2_fz.x, p2_fz.y) if hasattr(p2_fz, 'x') else tuple(p2_fz)

                        line_rect = fitz.Rect(p1, p2)
                        if overlaps_text(line_rect):
                            continue

                        length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        
                        # Only add long gray lines as walls
                        if length >= MIN_WALL_LENGTH_PDF and is_gray_stroke:
                            wall_segments_pdf.append((p1, p2))
                            gray_lines_found += 1

                elif item_type == 're':  # Rectangle (potentially a filled wall)
                    if len(points) >= 1:
                        rect = points[0]
                        if isinstance(rect, fitz.Rect):
                            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                        else:
                            x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]

                        rect_obj = fitz.Rect(x0, y0, x1, y1)

                        if overlaps_text(rect_obj):
                            continue

                        width = abs(x1 - x0)
                        height = abs(y1 - y0)

                        if width > 0 and height > 0:
                            aspect_ratio = max(width, height) / min(width, height)

                            if aspect_ratio >= MIN_FILLED_WALL_ASPECT_RATIO and is_gray_fill:
                                length = max(width, height)
                                if length >= MIN_WALL_LENGTH_PDF:
                                    # Add the two longer sides as wall segments
                                    if width > height:
                                        wall_segments_pdf.append(((x0, y0), (x1, y0)))
                                        wall_segments_pdf.append(((x0, y1), (x1, y1)))
                                    else:
                                        wall_segments_pdf.append(((x0, y0), (x0, y1)))
                                        wall_segments_pdf.append(((x1, y0), (x1, y1)))
                                    gray_rects_found += 1

        print(f"       Found {gray_lines_found} gray line segments (long walls).")
        print(f"       Found {gray_rects_found} gray elongated rectangles (filled walls).")

        # --- HATCHING DETECTION LOGIC REMOVED ---

        # Combine detected filled rectangles and hatched regions
        final_wall_segments = wall_segments_pdf

        # Remove duplicates
        unique_segments = set()
        final_processed_wall_segments = []

        for p1, p2 in final_wall_segments:
            # Create canonical representation (p1 is always top-left-ish)
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                p1, p2 = p2, p1
                
            pt1_tuple = (round(p1[0], 1), round(p1[1], 1))
            pt2_tuple = (round(p2[0], 1), round(p2[1], 1))
            segment_tuple = (pt1_tuple, pt2_tuple)

            if segment_tuple not in unique_segments:
                final_processed_wall_segments.append((p1, p2))
                unique_segments.add(segment_tuple)

        print(f"       Kept {len(final_processed_wall_segments)} final unique wall segments for annotation.")
        doc.close()

    except Exception as e:
        print(f"Error detecting walls from PDF: {e}")
        traceback.print_exc()
        final_processed_wall_segments = []  # Ensure it returns an empty list on error

    return final_processed_wall_segments
# --- END Wall Detection ---


# --- Generate Distinct Colors ---
def generate_distinct_colors(n):
    """Generates n visually distinct colors."""
    colors = []
    num_to_generate = max(n, 10)
    for i in range(num_to_generate):
        hue = i / num_to_generate
        lightness = 0.6 + (i % 3) * 0.1; saturation = 0.8 + (i % 2) * 0.1
        try:
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(tuple(max(0.0, min(1.0, c)) for c in rgb))
        except: colors.append((1.0, 0.0, 0.0))
    final_colors = colors[:n] if n <= num_to_generate else colors * (n // num_to_generate) + colors[:n % num_to_generate]
    if not final_colors: final_colors.append((1.0, 0.0, 0.0))
    return final_colors


# --- ROOM BOUNDARY DETECTION REMOVED ---


# --- PDF Creation Function ---
def create_annotated_pdf(input_pdf, found_rooms, found_doors, found_windows, wall_segments_pdf, output_pdf):
    """
    Create annotated PDF: highlight rooms, doors, windows, and draw detected walls.
    """
    doc = None
    try: doc = fitz.open(input_pdf)
    except Exception as e: print(f"Error opening PDF '{input_pdf}': {e}"); return

    door_label_color = (0, 0, 1); window_label_color = (0, 0.7, 0); text_color = (0, 0, 0); text_bg_color = (1, 1, 0.8)
    num_rooms = len(found_rooms); room_colors = generate_distinct_colors(max(num_rooms, 1))
    room_color_map = {room_num: room_colors[i % len(room_colors)] for i, room_num in enumerate(sorted(found_rooms.keys()))}
    label_scale_factor = 72.0 / IMAGE_DPI

    page_idx = 0
    if page_idx >= len(doc): print(f"Error: PDF page index {page_idx} out of range."); return
    page = doc[page_idx]
    print(f"\nAnnotating PDF page {page_idx + 1}...")

    # --- 0. ROOM FILLING LOGIC REMOVED ---

    # --- 1. Draw Detected Walls ---
    print(f"   Drawing {len(wall_segments_pdf)} detected wall segments (Red)...")
    shape = page.new_shape()
    walls_drawn = 0
    for p1, p2 in wall_segments_pdf:
        try:
            # Ensure points are valid numbers before creating fitz.Point
            if all(isinstance(coord, (int, float)) for coord in p1 + p2):
                # Check distance again to avoid zero-length lines
                if math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) > 0.1:
                    shape.draw_line(fitz.Point(p1), fitz.Point(p2))
                    walls_drawn += 1
            # else: print(f"       Warning: Skipping wall segment with invalid coords: {p1}-{p2}")
        except Exception as draw_e: print(f"       Warning: Could not draw wall segment {p1}-{p2}: {draw_e}")
    
    shape.finish(color=WALL_LINE_COLOR_PDF, width=WALL_LINE_WIDTH_PDF)
    
    try:
        if walls_drawn > 0:
            shape.commit()
            print(f"   Successfully committed {walls_drawn} wall segments.")
        # else: print("   No valid wall segments to commit.")
    except Exception as commit_e: print(f"       Warning: Could not commit wall shapes: {commit_e}")

    # --- 2. Add Labels and Highlights ---
    def add_label_and_highlight(page, bbox, label, highlight_color, text_bg_color, current_scale_factor):
        # (Keep the robust label adding function)
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox)): return None
        try: x_img, y_img, w_img, h_img = map(float, bbox); assert w_img > 0 and h_img > 0
        except: return None
        x0 = x_img * current_scale_factor; y0 = y_img * current_scale_factor
        x1 = x0 + (w_img * current_scale_factor); y1 = y0 + (h_img * current_scale_factor)
        page_width, page_height = page.rect.width, page.rect.height
        x0_c = max(0, min(x0, page_width)); y0_c = max(0, min(y0, page_height)); x1_c = max(x0_c, min(x1, page_width)); y1_c = max(y0_c, min(y1, page_height))
        highlight_added = False; highlight_rect = None
        if x1_c > x0_c and y1_c > y0_c:
            highlight_rect = fitz.Rect(x0_c, y0_c, x1_c, y1_c)
            try: highlight = page.add_highlight_annot(highlight_rect); highlight.set_colors(stroke=highlight_color); highlight.set_opacity(0.4); highlight.update(); highlight_added = True
            except: pass
        font_size = 8; padding = 2; text_anchor_x = x0; text_anchor_y = y0; text_y0 = text_anchor_y - font_size - (padding * 2); text_x0 = text_anchor_x
        try: text_width = fitz.get_text_length(label, fontsize=font_size)
        except: text_width = len(label) * font_size * 0.6
        text_x1 = text_x0 + text_width + (padding * 2); text_y1 = text_y0 + font_size + (padding * 2); text_rect = fitz.Rect(text_x0, text_y0, text_x1, text_y1)
        if text_rect.y0 < padding: text_rect.y0 = y1 + padding; text_rect.y1 = text_rect.y0 + font_size + (padding * 2)
        text_rect.x0 = max(padding, min(text_rect.x0, page_width - text_width - padding * 2)); text_rect.x1 = text_rect.x0 + text_width + (padding * 2)
        text_rect.y0 = max(padding, min(text_rect.y0, page_height - font_size - padding * 2)); text_rect.y1 = text_rect.y0 + font_size + (padding * 2)
        text_added = False
        if text_rect.x1 > text_rect.x0 and text_rect.y1 > text_rect.y0:
            try: text_annot = page.add_freetext_annot(text_rect, label, fontsize=font_size, text_color=text_color, fill_color=text_bg_color, align=fitz.TEXT_ALIGN_LEFT); text_annot.set_opacity(0.85); text_annot.update(); text_added = True
            except: pass
        if highlight_added or text_added: return (x0, y0)
        else: return None

    print("   Adding item labels...")
    items_added = 0
    for items_dict, category, color in [(found_doors, "DOOR", door_label_color), (found_windows, "WINDOW", window_label_color)]:
        for item_mark, item_info in items_dict.items():
            bbox = item_info.get('bbox')
            if not bbox: continue
            label = f"{category} {item_mark} ({item_info.get('shape', '?')})"
            coords = add_label_and_highlight(page, bbox, label, color, text_bg_color, label_scale_factor)
            if coords: items_added += 1
    for room_num, item_info in found_rooms.items():
        bbox = item_info.get('bbox')
        if not bbox: continue
        label = f"ROOM {room_num}{': ' + item_info.get('name', 'N/A') if item_info.get('name') else ''} ({item_info.get('shape', '?')})"
        room_highlight_color = room_color_map.get(room_num, (1, 0, 0))
        coords = add_label_and_highlight(page, bbox, label, room_highlight_color, text_bg_color, label_scale_factor)
        if coords: items_added += 1
    print(f"   Added {items_added} labels/highlights.")

    # --- Save PDF ---
    try:
        if items_added == 0 and not wall_segments_pdf: print("\nWarning: No annotations were added.")
        doc.save(output_pdf, garbage=4, deflate=True, clean=True)
        print(f"\nAnnotated PDF saved successfully to: {output_pdf}")
    except Exception as e: print(f"Error saving PDF '{output_pdf}': {e}")
    finally:
        if doc:
            try: doc.close()
            except: pass
# --- END PDF Creation ---


def main():
    """Main function."""
    try: script_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError: script_dir = os.getcwd(); print(f"Warn: Using CWD: {script_dir}")
    
    # Define file paths
    schedule_path = os.path.join(script_dir, 'schedule.json')
    ocr_data_path = os.path.join(script_dir, 'logs_ocr_data.json')
    input_pdf = os.path.join(script_dir, 'plan.pdf')
    output_pdf = os.path.join(script_dir, 'plan_rooms.pdf')  # Keep this filename

    print("=" * 60 + "\nSchedule Item & Gray Wall Annotation Script\n" + "=" * 60)  # Updated title
    print("\nLoading data files...")
    schedule = load_schedule(schedule_path);
    if not schedule: print("Exiting: Failed schedule load."); return
    ocr_data = load_ocr_data(ocr_data_path);  # Filtered to page 1
    if not ocr_data: print("Exiting: Failed OCR data load for page 1."); return

    print("\nExtracting marks from schedule...")
    room_marks = extract_room_marks(schedule)
    door_marks, window_marks = extract_door_window_marks(schedule)
    print(f"   Found {len(room_marks)} rooms, {len(door_marks)} doors, {len(window_marks)} windows in schedule.")

    print("\nSearching OCR results for schedule marks (Page 1 only)...")
    found_rooms = find_rooms_in_ocr(ocr_data, room_marks)
    found_doors = find_simple_marks_in_ocr(ocr_data, door_marks, "Door")
    found_windows = find_simple_marks_in_ocr(ocr_data, window_marks, "Window")

    print(f"\n--- Found Items Summary (Page 1) ---")
    print(f"- Rooms: {len(found_rooms)}, Doors: {len(found_doors)}, Windows: {len(found_windows)}")
    print("-------------------------")

    # --- Detect GRAY Walls from PDF Vector Data, Excluding Text ---
    wall_segments_pdf = []
    if not os.path.exists(input_pdf):
        print(f"Error: Input PDF '{input_pdf}' not found. Cannot detect walls.")
    else:
        print("\nDetecting GRAY walls from PDF vector data (excluding text regions)...")
        # Pass OCR data to exclude text areas
        wall_segments_pdf = detect_walls_from_pdf(input_pdf, page_num=0, ocr_data=ocr_data)

    # --- Detect Room Boundaries (REMOVED) ---

    # --- Create Annotated PDF ---
    if found_rooms or found_doors or found_windows or wall_segments_pdf:
        create_annotated_pdf(input_pdf, found_rooms, found_doors, found_windows, wall_segments_pdf, output_pdf)
    else: print("\nNo items or walls found to annotate. No PDF generated.")

    print("\n" + "=" * 66 + "\nScript finished.")
    if found_rooms or found_doors or found_windows or wall_segments_pdf: print(f"Output: {output_pdf}")
    print("=" * 66)


if __name__ == "__main__":
    main()