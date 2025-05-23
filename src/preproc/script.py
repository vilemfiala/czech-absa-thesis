import os
import time
import json
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# Nastavení
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-2.0-flash"

# CZ popisy sentimentů
TONE_CZ = {
    "negative": "velmi negativní",
    "neutral": "neutrální",
    "positive": "velmi pozitivní",
}
SENTIMENTS = tuple(TONE_CZ.keys())  # ("negative", "neutral", "positive")

ASPECTS_PER_ARTICLE = 3      # max. kolik aspektů extrahovat
MAX_PER_MIN = 500            # rychlostní limit (requests / min)
WORKERS = 60                 # paralelní vlákna (musí dělit MAX_PER_MIN)

# ─────────────────────────────────────────────────────────────────────────────
# Low‑level volání Geminia
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_call(prompt: str, max_tokens: int = 256) -> str:
    """Vyvolá model Gemini a vrátí čistý text."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=cfg,
    )
    return resp.text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# KROK 1 — detekce aspektů (max 3)
# ─────────────────────────────────────────────────────────────────────────────

def detect_aspects(text: str, n: int = ASPECTS_PER_ARTICLE):
    """Vrátí list aspektů (≤ n)."""
    prompt = f"""
Najdi {n} nejdůležitějších, vzájemně odlišných aspektů, které se týkají hlavního tématu následujícího textu. Aspekt je jednoduché podstatné jméno nebo fráze, která popisuje konkrétní vlastnost nebo aspekt daného tématu. Například: „kvalita produktu“, „cena“, „zákaznický servis“.
Napiš seznam aspektů, které jsou relevantní pro daný text. Nepoužívej žádné úvodní fráze ani vysvětlení, pouze seznam aspektů.
Odpověz POUZE jako JSON (bez komentářů!):
{{"aspects": ["…", "…", "…"]}}

TEXT:
\"\"\"{text}\"\"\"""".strip()

    raw = _gemini_call(prompt, max_tokens=128)
    try:
        # odstranění případného markdown formátování
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        raw = raw.replace("“", '"').replace("”", '"')
        data = json.loads(raw)

        # vynuceně vracíme přímo seznam aspektů
        aspects = data.get("aspects", [])
        if not isinstance(aspects, list):
            raise ValueError("Pole 'aspects' není seznam.")
        aspects = aspects[:n]
    except Exception as e:
        print(f"[DEBUG] Raw Gemini response:\n{raw}\n[ERROR] {e}")
        aspects = []
    return aspects


# ─────────────────────────────────────────────────────────────────────────────
# KROK 2 — shrnutí s různými sentimenty pro každý aspekt
# ─────────────────────────────────────────────────────────────────────────────

def summarize(text: str, aspect2sent: dict[str, str]) -> str:
    """Vygeneruje shrnutí, které zmíní všechny aspekty a u každého drží daný sentiment."""
    aspects_repr = ", ".join([f"{a} – {TONE_CZ[s]}" for a, s in aspect2sent.items()])
    prompt = f"""
Následující text se týká několika aspektů: {aspects_repr}.
Napiš souvislé shrnutí (max 200 slov), kde každý aspekt je zmíněn a popsán tónem, který mu byl přiřazen (pozitivní/negativní/neutrální). Snaž se o obecné shrnutí, kde jsou aspekty mimochodem zmíněny.
Nezapomeň na žádný aspekt a dbej, aby výsledný odstavec působil přirozeně. Pamatuj, že to má být shrnutí, ne analýza! I kdyby byl aspekt v textu zmíněň například s jiným tónem než zadáno, shrnutí by mělo být v souladu s tónem zadaným v úkolu a nikoli s tónem v textu.
TEXT:
\"\"\"{text}\"\"\"""".strip()
    print(f"[DEBUG] Shrnutí pro aspekty: {aspects_repr}")
    return _gemini_call(prompt, max_tokens=256)

# ─────────────────────────────────────────────────────────────────────────────
# Pomocná funkce — náhodné přiřazení sentimentů
# ─────────────────────────────────────────────────────────────────────────────

def random_sentiments(aspects: list[str]):
    """Vrátí dict {aspect: sentiment} s náhodným (možná opakovaným) sentimentem."""
    print(f"[DEBUG] Aspekty pro článek: {aspects}")
    return {a: random.choice(SENTIMENTS) for a in aspects}

# ─────────────────────────────────────────────────────────────────────────────
# Hlavní pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_excel(input_xlsx: str, output_xlsx: str):
    """Hlavní vstup→výstup: články → shrnutí + labely ABSA."""

    df = pd.read_excel(input_xlsx, engine="openpyxl")
    # df = df.head(5)

    text_col = "L" if "L" in df.columns else 11  # sloupec s textem článku
    id_col   = "A" if "A" in df.columns else 0   # interní ID / klíč

    # ── KROK 1: detekce aspektů ────────────────────────────────────────────
    det_tasks = [
        (
            row[id_col] if isinstance(id_col, str) else row.iloc[id_col],
            row[text_col] if isinstance(text_col, str) else row.iloc[text_col],
        )
        for _, row in df.iterrows()
        if not pd.isna(row[text_col] if isinstance(text_col, str) else row.iloc[text_col])
    ]

    aspects_per_id: dict = {}
    with tqdm(total=len(det_tasks), desc="Detekce aspektů", unit="req") as pbar:
        for chunk_i in range(0, len(det_tasks), MAX_PER_MIN):
            chunk = det_tasks[chunk_i : chunk_i + MAX_PER_MIN]
            start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(detect_aspects, text): aid for aid, text in chunk}
                for fut in as_completed(futures):
                    aid = futures[fut]
                    try:
                        aspects = fut.result()
                    except Exception as e:
                        print(f"CHYBA: {e}")
                        aspects = [f"<<CHYBA: {e}>>"]
                    aspects_per_id[aid] = aspects
                    pbar.update(1)

            # Rate‑limit na 200 req/min
            elapsed = time.perf_counter() - start
            if elapsed < 60:
                time.sleep(60 - elapsed)

    # ── KROK 2: shrnutí & labely ────────────────────────────────────────────
    sum_tasks = []  # (aid, text, aspect2sent)
    for aid, row in df.set_index(id_col if isinstance(id_col, str) else df.columns[id_col]).iterrows():
        text = row[text_col] if isinstance(text_col, str) else row.iloc[text_col]
        aspects = aspects_per_id.get(aid, [])
        if not aspects:
            continue
        aspect2sent = random_sentiments(aspects)
        sum_tasks.append((aid, text, aspect2sent))

    summaries: dict[int, str] = {}            # id → shrnutí (společné)
    aspect_sent_pairs: dict[int, list[tuple]] = {}  # id → [(aspect,sentiment), …]

    with tqdm(total=len(sum_tasks), desc="Generování shrnutí", unit="req") as pbar:
        for chunk_i in range(0, len(sum_tasks), MAX_PER_MIN):
            chunk = sum_tasks[chunk_i : chunk_i + MAX_PER_MIN]
            start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {
                    pool.submit(summarize, text, aspect2sent): (aid, aspect2sent)
                    for aid, text, aspect2sent in chunk
                }
                for fut in as_completed(futures):
                    aid, aspect2sent = futures[fut]
                    try:
                        summary = fut.result()
                    except Exception as e:
                        summary = f"<<CHYBA: {e}>>"
                    summaries[aid] = summary
                    aspect_sent_pairs[aid] = list(aspect2sent.items())
                    pbar.update(1)

            elapsed = time.perf_counter() - start
            if elapsed < 60:
                time.sleep(60 - elapsed)

    # ── Výstupní tabulka: jedna řádka = (id, aspekty+sentimenty, summary) ──
    rows = []
    for aid, pairs in aspect_sent_pairs.items():
        summary = summaries.get(aid, "")
        aspect_sents = " | ".join(f"{a} ({TONE_CZ.get(s, s)})" for a, s in pairs)
        rows.append({
            "id": aid,
            "aspekty_a_sentimenty": aspect_sents,
            "shrnutí": summary,
        })

    pd.DataFrame(rows).to_excel(output_xlsx, index=False, engine="openpyxl")

    print(f"✅ Hotovo! Uloženo {len(rows)} řádků → {output_xlsx}")


if __name__ == "__main__":
    process_excel("čeps.xlsx", "čeps_analysisv2.xlsx")
