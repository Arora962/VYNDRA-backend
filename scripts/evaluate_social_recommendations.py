#!/usr/bin/env python3
import json
from vyndra.social_recommendations import save_part5_reports, REPORT_DIR

if __name__ == "__main__":
    summary = save_part5_reports()
    print("Part 5 reports generated.")
    print(json.dumps(summary, indent=2))
    print(f"Reports saved to: {REPORT_DIR}")
