#!/usr/bin/env python3
"""Migrate lor_*/lens_* from category-level arrays to named per-concept properties.

v4 format (arrays):
    Memory: lor_culture = [1.2, -0.3, 0.0, ...]
    User:   lens_culture = [[8,2], [3,7], ...]

v5 format (named properties):
    Memory: lor_culture_individualism_collectivism = 1.2
    User:   lens_culture_individualism_collectivism = [8, 2]

This script:
1. Reads all Memory nodes with old lor_* array properties
2. Expands each array into named properties
3. Optionally removes old array properties (--remove-old)
4. Same for User nodes with lens_* array properties

Usage:
    # Dry run (print what would change)
    python scripts/migrate_named_properties.py --dry-run

    # Run against local FalkorDB
    python scripts/migrate_named_properties.py

    # Run and remove old array properties
    python scripts/migrate_named_properties.py --remove-old

    # Custom connection
    python scripts/migrate_named_properties.py --host localhost --port 16379 --graph memories
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automem.utils.lens_concepts import (
    CATEGORY_KEYS,
    LENS_CATEGORIES,
    lens_concept_property,
    lor_concept_property,
)


def migrate_memory_lor(graph, dry_run: bool, remove_old: bool) -> int:
    """Migrate Memory nodes: lor_<cat> arrays → lor_<cat>_<concept> floats."""
    # Find memories with old-format lor properties
    old_props = [f"lor_{cat}" for cat in CATEGORY_KEYS]
    return_clause = ", ".join(f"m.{p}" for p in old_props)
    query = f"MATCH (m:Memory) RETURN m.id, {return_clause}"

    result = graph.query(query)
    rows = getattr(result, "result_set", []) or []

    migrated = 0
    for row in rows:
        mid = row[0]
        set_clauses = []
        remove_clauses = []
        params = {"mid": mid}

        for i, cat in enumerate(CATEGORY_KEYS):
            val = row[i + 1]
            if val is None:
                continue

            # Parse array (may be list or JSON string)
            arr = val
            if isinstance(arr, (bytes, str)):
                try:
                    arr = json.loads(arr)
                except (ValueError, TypeError):
                    continue

            if not isinstance(arr, list):
                continue

            concepts = LENS_CATEGORIES[cat]
            for j, concept in enumerate(concepts):
                if j >= len(arr):
                    break
                lor_val = arr[j]
                if not isinstance(lor_val, (int, float)) or lor_val == 0.0:
                    continue
                prop = lor_concept_property(concept)
                param_name = f"v_{cat}_{j}"
                set_clauses.append(f"m.{prop} = ${param_name}")
                params[param_name] = float(lor_val)

            if remove_old:
                remove_clauses.append(f"m.lor_{cat}")

        if not set_clauses:
            continue

        if dry_run:
            print(f"  Memory {mid}: would SET {len(set_clauses)} named properties")
            migrated += 1
            continue

        cypher = f"MATCH (m:Memory {{id: $mid}}) SET {', '.join(set_clauses)}"
        if remove_clauses:
            cypher += " REMOVE " + ", ".join(remove_clauses)
        graph.query(cypher, params)
        migrated += 1

    return migrated


def migrate_user_lens(graph, dry_run: bool, remove_old: bool) -> int:
    """Migrate User nodes: lens_<cat> arrays → lens_<cat>_<concept> [a,b] pairs."""
    old_props = [f"lens_{cat}" for cat in CATEGORY_KEYS]
    return_clause = ", ".join(f"u.{p}" for p in old_props)
    query = f"MATCH (u:User) RETURN u.id, {return_clause}"

    result = graph.query(query)
    rows = getattr(result, "result_set", []) or []

    migrated = 0
    for row in rows:
        uid = row[0]
        set_clauses = []
        remove_clauses = []
        params = {"uid": uid}

        for i, cat in enumerate(CATEGORY_KEYS):
            val = row[i + 1]
            if val is None:
                continue

            arr = val
            if isinstance(arr, (bytes, str)):
                try:
                    arr = json.loads(arr)
                except (ValueError, TypeError):
                    continue

            if not isinstance(arr, list):
                continue

            concepts = LENS_CATEGORIES[cat]
            for j, concept in enumerate(concepts):
                if j >= len(arr):
                    break
                ab_val = arr[j]
                if not isinstance(ab_val, (list, tuple)) or len(ab_val) != 2:
                    continue
                # Skip uniform priors [1,1]
                a, b = ab_val
                if a == 1.0 and b == 1.0:
                    continue
                prop = lens_concept_property(concept)
                param_name = f"v_{cat}_{j}"
                set_clauses.append(f"u.{prop} = ${param_name}")
                params[param_name] = list(ab_val)

            if remove_old:
                remove_clauses.append(f"u.lens_{cat}")

        if not set_clauses:
            continue

        if dry_run:
            print(f"  User {uid}: would SET {len(set_clauses)} named properties")
            migrated += 1
            continue

        cypher = f"MATCH (u:User {{id: $uid}}) SET {', '.join(set_clauses)}"
        if remove_clauses:
            cypher += " REMOVE " + ", ".join(remove_clauses)
        graph.query(cypher, params)
        migrated += 1

    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate lor/lens to named properties")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--password", default=None)
    parser.add_argument("--graph", default="memories")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remove-old", action="store_true",
                        help="Remove old array properties after migration")
    args = parser.parse_args()

    from falkordb import FalkorDB

    conn_params = {"host": args.host, "port": args.port}
    if args.password:
        conn_params["password"] = args.password
        conn_params["username"] = "default"

    db = FalkorDB(**conn_params)
    graph = db.select_graph(args.graph)

    prefix = "[DRY RUN] " if args.dry_run else ""

    print(f"{prefix}Migrating Memory lor arrays → named properties...")
    mem_count = migrate_memory_lor(graph, args.dry_run, args.remove_old)
    print(f"{prefix}Migrated {mem_count} Memory nodes")

    print(f"\n{prefix}Migrating User lens arrays → named properties...")
    user_count = migrate_user_lens(graph, args.dry_run, args.remove_old)
    print(f"{prefix}Migrated {user_count} User nodes")

    print(f"\n{prefix}Done. Total: {mem_count} memories + {user_count} users")
    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")
    if not args.remove_old and not args.dry_run:
        print("\nOld array properties kept. Re-run with --remove-old to clean up.")


if __name__ == "__main__":
    main()
