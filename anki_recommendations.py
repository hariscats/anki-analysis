import json
import requests
import pandas as pd

ANKI_CONNECT_URL = "http://localhost:8765"


def invoke(action, **params):
    return requests.post(ANKI_CONNECT_URL, json={"action": action, "version": 6, "params": params}).json()


def fetch_reviewed_cards():
    card_ids = invoke("findCards", query="rated:1")
    ids = card_ids.get("result", [])
    if not ids:
        return pd.DataFrame(columns=["Question", "Answer", "Ease", "Interval", "Lapses", "Deck"])
    info = invoke("cardsInfo", cards=ids).get("result", [])
    records = []
    for card in info:
        fields = card.get("fields", {})
        question = fields.get("Front", {}).get("value") or next(iter(fields.values()))["value"] if fields else ""
        answer = fields.get("Back", {}).get("value") if "Back" in fields else ""
        records.append({
            "Question": question,
            "Answer": answer,
            "Ease": card.get("factor"),
            "Interval": card.get("interval"),
            "Lapses": card.get("lapses"),
            "Deck": card.get("deckName")
        })
    return pd.DataFrame(records)


def find_struggle_cards(df):
    return df[(df["Lapses"] >= 2) | (df["Ease"] < 2000)]


def group_and_sort(df):
    grouped = {}
    for _, row in df.iterrows():
        deck = row["Deck"]
        grouped.setdefault(deck, []).append(row)
    for deck in grouped:
        grouped[deck] = sorted(grouped[deck], key=lambda r: (-r["Lapses"], r["Ease"]))
    return grouped


def display_recommendations(grouped):
    print("Areas You Should Revisit or Explore Deeper:\n")
    for deck, cards in grouped.items():
        print(f"Deck: {deck}")
        for card in cards:
            question = card["Question"].replace('\n', ' ').strip()
            print(f"  - Flashcard: \"{question}\"")
            print(f"    - Lapses: {card['Lapses']}, Ease: {card['Ease']}")
        print()


def main():
    df = fetch_reviewed_cards()
    struggle_df = find_struggle_cards(df)
    grouped = group_and_sort(struggle_df)
    display_recommendations(grouped)


if __name__ == "__main__":
    main()
