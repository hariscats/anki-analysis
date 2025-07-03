# anki-analysis

This repository includes a helper script `anki_recommendations.py` for analyzing your reviewed Anki cards via the AnkiConnect API. It fetches card performance data, identifies cards you struggle with, and prints clear recommendations on what to revisit.

## Usage

1. Ensure [AnkiConnect](https://ankiweb.net/shared/info/2055492159) is installed and running in Anki.
2. Install the required Python dependencies (requests and pandas).
3. Run the script:

```bash
python anki_recommendations.py
```

The script outputs a list of cards with high lapse counts or low ease factors, grouped by deck.
