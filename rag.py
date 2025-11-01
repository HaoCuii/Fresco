import json
import os
import sys

import google.generativeai as genai


def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def create_system_prompt():
    return """
You are an Architectural Drawings Q&A assistant covering materials, dimensions, specs & layout relationships.


You will receive three JSON files per request:
Schedule file: all floors, door/window/material schedules.
Room map file: specific floor, coordinates of rooms, doors, walls.
OCR file: All of the text detected, may be out of order


Behavior / reasoning rules:
Be concise; state only verified facts.
Do not use filler phrases (“based on the information provided”, etc.).
If info is missing: answer with “Not enough information.” then provide where they could find that information
Always verify spatial/logical relationships from the room map. 
If something about location is asked, always use the coordinate system and when responding never use coordinates, always say relative position from known points
Use exact IDs and correct architectural terms when referencing doors, rooms, walls.
If two sources conflict: trust room map for spatial data, schedule for materials/specs.
Do not invent names, details or assume beyond the data.
Don’t over-explain reasoning — just give the conclusion clearly.
Always read all 3 JSON files fully before answering.
""".strip()


class ArchitecturalQASession:
    """Session that caches the JSON data and keeps chat history."""

    def __init__(self, schedule_data, room_map_data, ocr_data, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")

        system_context = f"""{create_system_prompt()}

# CACHED ARCHITECTURAL DATA:

## Schedule Data:
{json.dumps(schedule_data, indent=2)}

## Room Map Data:
{json.dumps(room_map_data, indent=2)}

## OCR Data (use only when helpful):
{json.dumps(ocr_data, indent=2)}

---

Now answer questions based on the above data.
"""

        self.chat = self.model.start_chat(
            history=[
                {"role": "user", "parts": [system_context]},
                {
                    "role": "model",
                    "parts": [
                        "Understood. I have loaded the architectural data and will follow the rules."
                    ],
                },
            ]
        )

    def ask(self, question: str) -> str:
        response = self.chat.send_message(question)
        return response.text


def interactive_mode(schedule_path, room_map_path, ocr_path, api_key):
    print("Loading JSON files...")
    schedule_data = load_json_file(schedule_path)
    room_map_data = load_json_file(room_map_path)
    ocr_data = load_json_file(ocr_path)
    print("✓ Files loaded")

    print("Initializing session...")
    session = ArchitecturalQASession(schedule_data, room_map_data, ocr_data, api_key)
    print("✓ Session ready\n")

    print("Architectural Q&A")
    print("=" * 50)
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        question = input("Q: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            answer = session.ask(question)
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def single_question_mode(question, schedule_path, room_map_path, ocr_path, api_key):
    schedule_data = load_json_file(schedule_path)
    room_map_data = load_json_file(room_map_path)
    ocr_data = load_json_file(ocr_path)

    session = ArchitecturalQASession(schedule_data, room_map_data, ocr_data, api_key)

    try:
        answer = session.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    api_key = "AIzaSyAR9CNcnULu37eOhipFXRZupH19P-GcypY" 

    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not configured")
        sys.exit(1)

    if len(sys.argv) < 4:
        print("Usage: python rag.py <schedule.json> <room_map.json> <ocr.json> [question]")
        print("\nInteractive:")
        print("  python rag.py schedule.json room_map.json logs_ocr_data.json")
        print("\nSingle question:")
        print('  python rag.py schedule.json room_map.json logs_ocr_data.json "Which doors connect rooms 201 and 202?"')
        sys.exit(1)

    schedule_path = sys.argv[1]
    room_map_path = sys.argv[2]
    ocr_path = sys.argv[3]

    for path in (schedule_path, room_map_path, ocr_path):
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    if len(sys.argv) > 4:
        question = sys.argv[4]
        single_question_mode(question, schedule_path, room_map_path, ocr_path, api_key)
    else:
        interactive_mode(schedule_path, room_map_path, ocr_path, api_key)


if __name__ == "__main__":
    main()
