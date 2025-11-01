import json
import os
import sys
from pathlib import Path

import google.generativeai as genai
from PIL import Image


def analyze_schedule_images(image_paths, api_key):
    """
    Send three images to Gemini API and extract schedule information.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    images = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(Image.open(img_path))

    prompt = """
Analyze these three images and extract all schedule information (door schedule, window schedule, interior finish schedule, etc.).

Please return the data in JSON format with the following structure:
{
  "doorSchedule": {
    "groundFloor": [array of door objects],
    "firstFloor": [array of door objects],
    "secondFloor": [array of door objects],
    "notes": [array of note strings]
  },
  "interiorFinishSchedule": {
    "groundFloor": [array of room objects],
    "firstFloor": [array of room objects],
    "secondFloor": [array of room objects],
    "notes": [array of note strings],
    "abbreviations": [array of abbreviation objects]
  },
  "windowSchedule": {
    "groundFloor": [array of window objects],
    "firstFloor": [array of window objects],
    "secondFloor": [array of window objects],
    "notes": [array of note strings]
  }
}

For door objects, use this structure:
{
  "MARK": "string or null",
  "TYPE": "string or null",
  "W X H": "string or null",
  "FINISH OPEN": "string or null",
  "DOOR": {
    "MATERIAL": "string or null",
    "FINISH": "string or null"
  },
  "FRAME": {
    "MATERIAL": "string or null",
    "FINISH": "string or null"
  },
  "HDWR": "string or null",
  "COMMENT": "string or null"
}

For interior finish objects, use this structure:
{
  "number": "string or null",
  "NAME": "string or null",
  "BASE": "string or null",
  "WALLS": {
    "NORTH": "string or null",
    "EAST": "string or null",
    "SOUTH": "string or null",
    "WEST": "string or null"
  },
  "CEILING": "string or null",
  "COMMENT": "string or null"
}

For window objects, use this structure:
{
  "MARK": "string or null",
  "TYPE": "string or null",
  "W X H": "string or null",
  "GLAZING": {
    "U-FACT": "string or null",
    "SHGC": "string or null"
  } or null,
  "SHADE MATERIAL": "string or null",
  "FINISH": "string or null",
  "NOTE": "string or null",
  "COMMENTS": "string or null"
}

For abbreviation objects:
{
  "ABBR": "string",
  "DESCRIPTION": "string"
}

IMPORTANT RULES:
1. If a field is not present in the images, set it to null (NOT the string "na")
2. If you find additional fields not in this structure, add them to the appropriate objects
3. Extract ALL rows from the schedules, even if some fields are empty
4. Preserve exact values as they appear in the images
5. Make sure to extract all notes from the bottom of each schedule
6. Return ONLY valid JSON, no additional text or markdown

Please analyze the images carefully and extract all visible schedule data.
""".strip()

    content = images + [prompt]
    response = model.generate_content(content)
    response_text = response.text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text:\n{response_text}")
        raise


def save_schedule(schedule_data, output_path="schedule.json"):
    with open(output_path, "w") as f:
        json.dump(schedule_data, f, indent=2)
    print(f"Schedule saved to {output_path}")


def main():
    api_key = "AIzaSyAR9CNcnULu37eOhipFXRZupH19P-GcypY"  #Exposed lol

    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not configured")
        print("Please set it with: export GOOGLE_API_KEY='your-api-key-here'")
        print("Or on Windows: set GOOGLE_API_KEY=your-api-key-here")
        sys.exit(1)

    if len(sys.argv) < 4:
        print("Usage: python schedule.py <image1_path> <image2_path> <image3_path> [output_path]")
        print("\nExamples:")
        print("  python schedule.py door_schedule.png window_schedule.png finish_schedule.png")
        print("  python schedule.py img1.jpg img2.jpg img3.jpg custom_output.json")
        print("\nNote: Make sure to set GOOGLE_API_KEY environment variable first")
        sys.exit(1)

    image_paths = sys.argv[1:4]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "schedule_output.json"

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            sys.exit(1)

    print("Analyzing images:")
    for i, img_path in enumerate(image_paths, 1):
        print(f"  {i}. {img_path}")

    print("\nSending images to Gemini API...")

    try:
        schedule_data = analyze_schedule_images(image_paths, api_key)
        print("Successfully analyzed images and extracted schedule data")

        save_schedule(schedule_data, output_path)
        print(f"Schedule saved to {output_path}")

        print("\nSchedule Summary:")
        if "doorSchedule" in schedule_data:
            door_count = sum(
                len(schedule_data["doorSchedule"].get(floor, []))
                for floor in ["groundFloor", "firstFloor", "secondFloor"]
            )
            print(f"  - Doors: {door_count}")

        if "windowSchedule" in schedule_data:
            window_count = sum(
                len(schedule_data["windowSchedule"].get(floor, []))
                for floor in ["groundFloor", "firstFloor", "secondFloor"]
            )
            print(f"  - Windows: {window_count}")

        if "interiorFinishSchedule" in schedule_data:
            room_count = sum(
                len(schedule_data["interiorFinishSchedule"].get(floor, []))
                for floor in ["groundFloor", "firstFloor", "secondFloor"]
            )
            print(f"  - Rooms: {room_count}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
