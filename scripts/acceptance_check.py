#!/usr/bin/env python3
"""Acceptance checks for completed VYNDRA parts; exits non-zero on failed checks."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check(condition: bool, message: str, failures: list[str]):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)

def part1(failures):
    check((ROOT / "scripts" / "acceptance_check.py").is_file(), "Acceptance-check script exists", failures)
    check((ROOT / "vyndra" / "photo_api.py").is_file(), "Photo API module exists", failures)
    check((ROOT / "vyndra" / "portions.py").is_file(), "Serving-size module exists", failures)
    check((ROOT / "scripts" / "train_food_intent.py").is_file(), "Intent-model training script exists", failures)
    check((ROOT / "data" / "processed" / "reference_meals_template.csv").is_file(), "Reference-meal template exists", failures)
    check((ROOT / "docs" / "API_REFERENCE.md").is_file(), "API reference exists", failures)
    try:
        import app
        routes = {rule.rule for rule in app.app.url_map.iter_rules()}
        check("/upload" in routes, "POST /upload alias is registered", failures)
        check("/api/meals/analyze" in routes, "POST /api/meals/analyze is registered", failures)
        check("/api/chat/send" in routes, "POST /api/chat/send is registered", failures)
        check("/api/chat/send-meal" in routes, "POST /api/chat/send-meal is registered", failures)
        check("/api/insights/<user_id>" in routes, "GET /api/insights/<user_id> is registered", failures)
        check("/api/users" in routes, "User create/list routes are registered", failures)
        check("/api/meals" in routes, "Meal list route is registered", failures)
        check("/api/chat/<conversation_id>" in routes, "Conversation history route is registered", failures)
        check("/api/social/influencers" in routes, "Social influencers route is registered", failures)
        check("/api/social/communities" in routes, "Social communities route is registered", failures)
        check("/api/foods/<path:food_name>/insights" in routes, "Food insights route is registered", failures)
        check(sum(1 for rule in app.app.url_map.iter_rules() if rule.rule == "/api/chat/analyze") == 1,
              "Legacy chat analyze route is registered exactly once", failures)
        check(app.app.config.get("MAX_CONTENT_LENGTH", 0) >= 8 * 1024 * 1024,
              "Flask request limit permits the documented 8 MB image upload", failures)
    except Exception as exc:
        check(False, f"Application imports and routes can be inspected ({type(exc).__name__})", failures)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=range(1, 11), help="Check a single project part")
    parser.add_argument("--all", action="store_true", help="Run all acceptance checks available in the repository")
    args = parser.parse_args()
    if not args.part and not args.all:
        parser.error("choose --part N or --all")
    failures = []
    if args.part not in (None, 1) and not args.all:
        print(f"[NOT IMPLEMENTED] Acceptance criteria for Part {args.part} are added in a later delivery.")
        return 2
    if args.part == 1 or args.all:
        part1(failures)
    # Later project parts add their own checks here. --all intentionally checks
    # only implemented criteria and will be expanded in Parts 2–5.
    if failures:
        print(f"\nAcceptance checks failed: {len(failures)}")
        return 1
    print("\nAcceptance checks passed for the selected implemented scope.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
