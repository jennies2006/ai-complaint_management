"""
Run ai_module/tests/sample_cases.json against predict_complaint().
Usage: python -m ai_module.tests.run_sample_cases
"""
import json
import sys
from pathlib import Path

from ai_module.predict import predict_complaint

CASES_PATH = Path(__file__).parent / "sample_cases.json"
PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]


def main():
    cases = json.loads(CASES_PATH.read_text())
    passed = 0
    failed = 0
    pair_results = {}

    for c in cases:
        text = c.get("text", "")
        result = predict_complaint(text)

        ok = True
        reasons = []

        if "expect_status" in c:
            if result["status"] != c["expect_status"]:
                ok = False
                reasons.append(f"status {result['status']} != {c['expect_status']}")

        if "expect_category" in c and result["status"] == "ok":
            if result["category"] != c["expect_category"]:
                ok = False
                reasons.append(f"category {result['category']} != {c['expect_category']}")

        if "expect_priority" in c and result["status"] == "ok":
            if result["priority"] != c["expect_priority"]:
                ok = False
                reasons.append(f"priority {result['priority']} != {c['expect_priority']}")

        if "expect_urgency_contains" in c and result["status"] == "ok":
            for sig in c["expect_urgency_contains"]:
                if sig not in result["urgency_signals"]:
                    ok = False
                    reasons.append(f"missing urgency signal '{sig}'")

        if "pair_group" in c and result["status"] == "ok":
            pair_results.setdefault(c["pair_group"], []).append(
                (c["id"], c.get("expect_priority_relative"), result["priority"])
            )

        if ok:
            passed += 1
            print(f"✅ {c['id']}")
        else:
            failed += 1
            print(f"❌ {c['id']}: {', '.join(reasons)}")
            print(f"    got: {result}")

    print("\n" + "=" * 60)
    print("TONE PAIR ORDERING")
    print("=" * 60)
    for group, entries in pair_results.items():
        lower = next((p for _, rel, p in entries if rel == "lower"), None)
        higher = next((p for _, rel, p in entries if rel == "higher"), None)
        if lower and higher:
            if PRIORITY_ORDER.index(higher) > PRIORITY_ORDER.index(lower):
                print(f"✅ {group:<12} {lower} < {higher}")
                passed += 1
            else:
                print(f"❌ {group:<12} {lower} !< {higher}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"RESULT: {passed} passed, {failed} failed ({len(cases)} cases)")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
