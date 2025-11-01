# Fresco – Setup & Run

## 1. Prerequisites
- Install **Tesseract OCR**  
  - Windows default: `C:\Program Files\Tesseract-OCR\tesseract.exe`  
- `python -m pip install -r requirements.txt`

## 2. Step 1 – Extract Schedules (Gemini)
Run this with your **three** schedule images (door/window/finish):
- `python schedule.py img1.png img2.png img3.png`
- Output: `schedule.json`

## 3. Step 2 – OCR the Plan
- Make sure your drawing is `plan.pdf` in the same folder
- Run: `python textdetect.py`
- Output: `logs_ocr_data.json` and  `plan_text.pdf`

## 4. Step 3 – Build Room / Wall Map
- Run: `python room.py`
- Reads: `schedule.json`, `logs_ocr_data.json`, `plan.pdf`
- Output: `room_map.json` and  `plan_room.pdf`

## 5. Step 4 – Architectural Q&A
- Run: `python rag.py schedule.json room_map.json logs_ocr_data.json`

## Notes
- Gemini API is not the best, ChaptGPT works better. Try pasting the three desired files along with the prompt into ChatGPT.
- Questions about location could be improved. With multiple follow up questions the LLM will usually be able to correctly answer the prompt.
