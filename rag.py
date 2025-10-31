import google.generativeai as genai
import json
import os
import sys


def load_json_file(file_path):
    """Load and parse a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def create_system_prompt():
    """Create the system prompt with the architectural Q&A instructions"""
    return """You are an Architectural Drawings Q&A assistant covering materials, dimensions, specs & layout relationships.

You will receive three JSON files per request:
1. Schedule file: all floors, door/window/material schedules.
2. Room map file: specific floor, coordinates of rooms, doors, walls.
3. OCR file: extracted drawing text (may contain noise) — use only when helpful.

Behavior / reasoning rules:
- Be concise; state only verified facts.
- Do not use filler phrases ("based on the information provided", etc.).
- If info is missing: answer with "Not enough information."
- Always verify spatial/logical relationships (door connections, adjacency) from the room map.
- Use exact IDs and correct architectural terms when referencing doors, rooms, walls.
- If two sources conflict: trust room map for spatial data, schedule for materials/specs.
- Do not invent names, details or assume beyond the data.
- Don't over-explain reasoning — just give the conclusion clearly.

Example Behavior:

Q: What wall should I use for Carriage Gallery 1?
A: North wall: Wood (WD). East, South, West: Gypsum board (GB) with ½″ plywood backing.

Q: Which doors connect Carriage Gallery 1 and 2?
A: Door 204 and Door 205. Both are CO-type gypsum board doors, 7′-8″ × 12′-0″, with painted GB frames.

Q: Should I have a structural steel frame?
A: Not enough information."""


class ArchitecturalQASession:
    """A session that caches the JSON data and maintains conversation history"""

    def __init__(self, schedule_data, room_map_data, ocr_data, api_key):
        """Initialize the session with cached data"""
        # Configure Gemini API
        genai.configure(api_key=api_key)

        # Create model
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        # Create the system context with all JSON data (cached in conversation)
        self.system_context = f"""{create_system_prompt()}

# CACHED ARCHITECTURAL DATA:

## Schedule Data:
{json.dumps(schedule_data, indent=2)}

## Room Map Data:
{json.dumps(room_map_data, indent=2)}

## OCR Data (use only when helpful):
{json.dumps(ocr_data, indent=2)}

---

Now answer questions based on the above data."""

        # Initialize chat session with the data preloaded
        # This way the JSON is sent once and reused for all questions
        self.chat = self.model.start_chat(history=[
            {'role': 'user', 'parts': [self.system_context]},
            {'role': 'model', 'parts': ['Understood. I have loaded all the architectural data and will answer questions following the specified behavior rules.']}
        ])

    def ask(self, question):
        """Ask a question using the cached context"""
        response = self.chat.send_message(question)
        return response.text


def interactive_mode(schedule_path, room_map_path, ocr_path, api_key):
    """Run in interactive mode for multiple questions"""
    # Load JSON files once
    print("Loading JSON files...")
    schedule_data = load_json_file(schedule_path)
    room_map_data = load_json_file(room_map_path)
    ocr_data = load_json_file(ocr_path)
    print("✓ Files loaded successfully")

    # Create a cached session (data is sent once to Gemini)
    print("Initializing session with cached data...")
    session = ArchitecturalQASession(schedule_data, room_map_data, ocr_data, api_key)
    print("✓ Session initialized (JSON data cached)\n")

    print("Architectural Q&A System")
    print("=" * 50)
    print("Ask questions about the architectural drawings.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        # Get user question
        question = input("Q: ").strip()

        if not question:
            continue

        if question.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break

        try:
            # Query the cached session
            answer = session.ask(question)
            print(f"A: {answer}\n")

        except Exception as e:
            print(f"Error: {e}\n")


def single_question_mode(question, schedule_path, room_map_path, ocr_path, api_key):
    """Answer a single question and exit"""
    # Load JSON files
    schedule_data = load_json_file(schedule_path)
    room_map_data = load_json_file(room_map_path)
    ocr_data = load_json_file(ocr_path)

    # Create session with cached data
    session = ArchitecturalQASession(schedule_data, room_map_data, ocr_data, api_key)

    try:
        # Query the session
        answer = session.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main function"""
    # Check for API key
    api_key = "AIzaSyAR9CNcnULu37eOhipFXRZupH19P-GcypY"  # Your API key

    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not configured")
        sys.exit(1)

    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python rag.py <schedule.json> <room_map.json> <ocr.json> [question]")
        print("\nModes:")
        print("  Interactive mode (no question):")
        print("    python rag.py schedule.json room_map.json logs_ocr_data.json")
        print("\n  Single question mode:")
        print("    python rag.py schedule.json room_map.json logs_ocr_data.json \"What wall should I use for Carriage Gallery 1?\"")
        print("\nExamples:")
        print("  python rag.py schedule.json room_map.json logs_ocr_data.json")
        print("  python rag.py schedule.json room_map.json logs_ocr_data.json \"Which doors connect rooms 201 and 202?\"")
        sys.exit(1)

    schedule_path = sys.argv[1]
    room_map_path = sys.argv[2]
    ocr_path = sys.argv[3]

    # Check if files exist
    for path in [schedule_path, room_map_path, ocr_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    # Determine mode based on whether a question was provided
    if len(sys.argv) > 4:
        # Single question mode
        question = sys.argv[4]
        single_question_mode(question, schedule_path, room_map_path, ocr_path, api_key)
    else:
        # Interactive mode
        interactive_mode(schedule_path, room_map_path, ocr_path, api_key)


if __name__ == "__main__":
    main()
