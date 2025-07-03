import json
import requests
import pandas as pd
import re
from datetime import datetime
import os

ANKI_CONNECT_URL = "http://localhost:8765"

def invoke(action, params=None):
    """Make a request to AnkiConnect"""
    if params is None:
        params = {}
    
    request_data = {
        "action": action,
        "version": 6,
        "params": params
    }
    
    response = requests.post(ANKI_CONNECT_URL, json=request_data)
    response.raise_for_status()
    return response.json()

def fetch_reviewed_cards():
    """Fetch cards reviewed in the last day"""
    # Get card IDs for cards reviewed in the last day
    card_ids = invoke("findCards", {"query": "rated:1"})["result"]
    
    if not card_ids:
        return pd.DataFrame()
    
    # Get card info
    cards_info = invoke("cardsInfo", {"cards": card_ids})["result"]
    
    if not cards_info:
        return pd.DataFrame()
    
    # Get notes info for the cards - check for correct note ID key
    note_ids = []
    for card in cards_info:
        # Try different possible keys for note ID
        if "nid" in card:
            note_ids.append(card["nid"])
        elif "note" in card:
            note_ids.append(card["note"])
        elif "noteId" in card:
            note_ids.append(card["noteId"])
    
    if not note_ids:
        return pd.DataFrame()
    
    notes_info = invoke("notesInfo", {"notes": note_ids})["result"]
    
    # Create a mapping of note ID to note info
    note_map = {}
    for note in notes_info:
        # Try different possible keys for note ID
        if "noteId" in note:
            note_map[note["noteId"]] = note
        elif "nid" in note:
            note_map[note["nid"]] = note
        elif "id" in note:
            note_map[note["id"]] = note
    
    # Process the data
    processed_cards = []
    for card in cards_info:
        # Get the note ID using the correct key
        note_id = None
        if "nid" in card:
            note_id = card["nid"]
        elif "note" in card:
            note_id = card["note"]
        elif "noteId" in card:
            note_id = card["noteId"]
        
        if note_id is None:
            continue
            
        note = note_map.get(note_id)
        if note:
            # Get question and answer (first two fields typically)
            fields = note.get("fields", {})
            field_values = list(fields.values()) if fields else []
            
            question = field_values[0]["value"] if len(field_values) > 0 else ""
            answer = field_values[1]["value"] if len(field_values) > 1 else ""
            
            # Clean HTML tags
            question = re.sub(r'<[^>]+>', '', question)
            answer = re.sub(r'<[^>]+>', '', answer)
            
            processed_cards.append({
                "Question": question,
                "Answer": answer,
                "Deck": card.get("deckName", "Unknown"),
                "Lapses": card.get("lapses", 0),
                "Ease": card.get("factor", 2500),
                "Interval": card.get("ivl", 1),
                "Reviews": card.get("reps", 0)
            })
    
    return pd.DataFrame(processed_cards)

def find_struggle_cards(df):
    """Identify cards that are struggling based on multiple criteria"""
    # Define struggle criteria with different severity levels
    severe_struggle = (df["Lapses"] >= 5) | (df["Ease"] < 1500)
    moderate_struggle = (df["Lapses"] >= 3) | (df["Ease"] < 1800)
    mild_struggle = (df["Lapses"] >= 2) | (df["Ease"] < 2000)
    
    # Add struggle severity column
    df["Struggle_Level"] = "None"
    df.loc[mild_struggle, "Struggle_Level"] = "Mild"
    df.loc[moderate_struggle, "Struggle_Level"] = "Moderate"
    df.loc[severe_struggle, "Struggle_Level"] = "Severe"
    
    # Calculate struggle score (higher = more problematic)
    # Score based on lapses (weighted heavily) and ease factor
    df["Struggle_Score"] = (df["Lapses"] * 3) + ((2500 - df["Ease"]) / 100)
    
    # Only return cards with some struggle
    struggle_cards = df[df["Struggle_Level"] != "None"].copy()
    
    return struggle_cards

def export_struggle_report(df, filename=None):
    """Export struggle cards to CSV"""
    if filename is None:
        filename = f"anki_struggle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Sort by struggle score (highest first)
    df_sorted = df.sort_values("Struggle_Score", ascending=False)
    
    # Reorder columns for better readability
    column_order = ["Struggle_Level", "Struggle_Score", "Question", "Answer", 
                   "Deck", "Lapses", "Ease", "Interval", "Reviews"]
    df_export = df_sorted[column_order]
    
    df_export.to_csv(filename, index=False)
    return filename

def display_struggle_report(df):
    """Display struggle cards report"""
    if df.empty:
        print("🎉 Great! No struggling cards found.")
        return
    
    print("=== ANKI STRUGGLE REPORT ===\n")
    
    # Summary statistics
    total_cards = len(df)
    severe_count = len(df[df["Struggle_Level"] == "Severe"])
    moderate_count = len(df[df["Struggle_Level"] == "Moderate"])
    mild_count = len(df[df["Struggle_Level"] == "Mild"])
    
    print("📊 SUMMARY:")
    print(f"   Total struggling cards: {total_cards}")
    print(f"   🔴 Severe struggles: {severe_count}")
    print(f"   🟡 Moderate struggles: {moderate_count}")
    print(f"   🟢 Mild struggles: {mild_count}")
    print(f"   Average lapses: {df['Lapses'].mean():.1f}")
    print(f"   Average ease: {df['Ease'].mean():.0f}")
    
    # Top struggling cards
    print(f"\n🎯 TOP {min(20, len(df))} MOST PROBLEMATIC CARDS:")
    print("-" * 100)
    
    df_sorted = df.sort_values("Struggle_Score", ascending=False)
    
    for i, (_, card) in enumerate(df_sorted.head(20).iterrows(), 1):
        question = card["Question"].replace('\n', ' ').strip()
        if len(question) > 60:
            question = question[:57] + "..."
        
        # Color coding for struggle level
        level_emoji = {"Severe": "🔴", "Moderate": "🟡", "Mild": "🟢"}
        emoji = level_emoji.get(card["Struggle_Level"], "⚪")
        
        print(f"{i:2d}. {emoji} {question}")
        print(f"     📊 Lapses: {card['Lapses']}, Ease: {card['Ease']}, "
              f"Score: {card['Struggle_Score']:.1f}")
        print(f"     📁 Deck: {card['Deck']}")
        print()
    
    # Deck breakdown
    print("\n📚 STRUGGLE BREAKDOWN BY DECK:")
    print("-" * 60)
    deck_stats = df.groupby("Deck").agg({
        "Struggle_Score": ["count", "mean"],
        "Lapses": "mean",
        "Ease": "mean"
    }).round(1)
    
    deck_stats.columns = ["Count", "Avg_Score", "Avg_Lapses", "Avg_Ease"]
    deck_stats = deck_stats.sort_values("Avg_Score", ascending=False)
    
    for deck, stats in deck_stats.iterrows():
        print(f"{deck[:30]:30} | Cards: {stats['Count']:3.0f} | "
              f"Score: {stats['Avg_Score']:5.1f} | "
              f"Lapses: {stats['Avg_Lapses']:4.1f} | "
              f"Ease: {stats['Avg_Ease']:6.0f}")

def main():
    print("🔍 Starting Anki Struggle Analysis...")
    
    # Test connection
    try:
        response = invoke("version")
        if response.get("result"):
            print(f"✅ Connected to AnkiConnect (version {response['result']})")
        else:
            print("❌ AnkiConnect error:", response.get("error"))
            return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure Anki is running and AnkiConnect is installed.")
        return
    
    # Fetch and analyze cards
    print("📊 Fetching reviewed cards...")
    df = fetch_reviewed_cards()
    
    if df.empty:
        print("No cards found. Make sure you've reviewed some cards recently.")
        return
    
    print(f"📈 Found {len(df)} reviewed cards")
    
    # Find struggle cards
    struggle_df = find_struggle_cards(df)
    print(f"⚠️  Identified {len(struggle_df)} cards that need attention")
    
    # Display results
    display_struggle_report(struggle_df)
    
    # Export to CSV
    if not struggle_df.empty:
        print("\n💾 Exporting struggle report...")
        filename = export_struggle_report(struggle_df)
        print(f"✅ Report saved as: {filename}")
    
    print(f"\n🎯 Analysis complete!")

if __name__ == "__main__":
    main()
