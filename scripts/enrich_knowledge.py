"""Enrich background_knowledge JSON files with type classification and reasoning.

Reads existing background_knowledge/*.json files.
For each item, calls nano (reasoning_effort=high) to:
  1. Classify: principle / pattern / insight
  2. Generate reasoning: why important, when to recall, hidden context

Outputs enriched JSON to background_knowledge/enriched/*.json

Usage:
  python scripts/enrich_knowledge.py                          # all files
  python scripts/enrich_knowledge.py --file taleb_deep.json   # single file
  python scripts/enrich_knowledge.py --dry-run                # preview only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "memory" / "background_knowledge"
OUTPUT_DIR = KNOWLEDGE_DIR / "enriched"

SYSTEM_PROMPT = """You enrich knowledge items for vector search embedding.

For each item you receive, produce:

1. **type**: Classify as exactly one of:
   - "principle" — a fundamental rule, law, or guideline that prescribes behavior
   - "pattern" — a recurring structure, approach, or phenomenon that describes what happens
   - "insight" — a specific observation, discovery, or realization that illuminates understanding

2. **reasoning**: Rich text (300-800 words) covering:
   - What this is fundamentally about (the deeper abstraction beyond the surface)
   - Why someone would need to recall this — what situation triggers its relevance
   - What hidden assumptions or counterintuitive aspects exist
   - What related concepts, keywords, or questions connect to this
   - What domains beyond the obvious this applies to (cross-domain transfer)
   - What common mistakes people make when applying or ignoring this

Write reasoning as flowing prose, not bullet points. Think deeply.
The reasoning will be embedded alongside the content for vector search —
the richer it is, the more diverse queries will match this knowledge.

Output JSON: {"type": "principle|pattern|insight", "reasoning": "..."}
JSON only, no markdown fences."""

USER_TEMPLATE = """---
{content}
---

Classify and generate reasoning for this knowledge item."""


def get_client():
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def enrich_item(client, content: str, model: str = "gpt-4.1-nano") -> dict:
    """Call nano with reasoning_effort=high to classify and generate reasoning."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(content=content)},
        ],
        temperature=0.0,
        max_tokens=3000,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return json.loads(raw)


def process_file(client, filepath: Path, dry_run: bool = False, model: str = "gpt-4.1-nano"):
    """Process a single background_knowledge JSON file."""
    with open(filepath) as f:
        items = json.load(f)

    if not isinstance(items, list):
        print(f"  SKIP {filepath.name}: not a JSON array")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name} ({len(items)} items)")
    print(f"{'='*60}")

    enriched = []
    for i, item in enumerate(items):
        content = item.get("content", "")
        if not content:
            continue

        # Extract title from content (format: "Author: Title. ...")
        title = content.split(".")[0] if "." in content else content[:80]

        print(f"  [{i+1}/{len(items)}] {title[:60]}...")

        if dry_run:
            enriched.append({**item, "type": "?", "reasoning": "(dry run)"})
            continue

        try:
            result = enrich_item(client, content, model=model)
            item_enriched = {
                **item,
                "type": result.get("type", "insight"),
                "reasoning": result.get("reasoning", ""),
            }
            enriched.append(item_enriched)
            print(f"    → type={item_enriched['type']}, reasoning={len(item_enriched['reasoning'])} chars")
            # Rate limit
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR: {e}")
            enriched.append({**item, "type": "insight", "reasoning": f"ERROR: {e}"})

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filepath.name
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path} ({len(enriched)} items)")
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich background knowledge with reasoning")
    parser.add_argument("--file", help="Process single file (e.g., taleb_deep.json)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    args = parser.parse_args()

    client = None if args.dry_run else get_client()

    if args.file:
        filepath = KNOWLEDGE_DIR / args.file
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        process_file(client, filepath, dry_run=args.dry_run, model=args.model)
    else:
        files = sorted(KNOWLEDGE_DIR.glob("*.json"))
        files = [f for f in files if f.name not in ("_to_store.json", "_bulk_store.py")]
        print(f"Found {len(files)} knowledge files")
        total_items = 0
        for filepath in files:
            result = process_file(client, filepath, dry_run=args.dry_run, model=args.model)
            if result:
                total_items += len(result)
        print(f"\nDone. Total: {total_items} items enriched.")


if __name__ == "__main__":
    main()
