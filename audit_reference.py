"""Run manually selected court-membership spot checks on the reference video."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from court import detect_court, point_on_court


def audit(video="1031(2).mp4", output=Path("outputs/reference")):
    specification = json.loads(Path(__file__).with_name("reference_cases.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    results = []
    try:
        for case in specification["cases"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, case["frame"])
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Frame {case['frame']} tidak terbaca.")
            mask, polygon, _ = detect_court(frame)
            result = {"frame": case["frame"], "checked_points": 0, "failures": []}
            if case.get("no_floor") and polygon:
                result["failures"].append("Expected no valid floor in closeup")
            if polygon:
                cv2.polylines(frame, [np.asarray(polygon, np.int32)], True, (255, 255, 0), 2)
            for key, expected in (("inside", True), ("outside", False)):
                for point in case[key]:
                    result["checked_points"] += 1
                    actual = point_on_court(mask, point)
                    if actual != expected:
                        result["failures"].append({"point": point, "expected_inside": expected})
                    color = (80, 220, 80) if actual == expected else (0, 0, 255)
                    cv2.circle(frame, tuple(point), 10, color, 3)
                    cv2.putText(frame, key, (point[0] + 12, max(20, point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
            cv2.imwrite(str(output / f"court-audit-{case['frame']:04d}.jpg"), frame)
            results.append(result)
    finally:
        capture.release()
    report = {"scope": specification["scope"], "cases": results,
              "checked_points": sum(r["checked_points"] for r in results),
              "failed_checks": sum(len(r["failures"]) for r in results)}
    (output / "court-spot-checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
