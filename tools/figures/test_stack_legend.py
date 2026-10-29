#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check rendered legend geometry. Run with Python 3 and Inkscape installed."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from make_figures import fig_stack


def legend_bounds(svg):
    """Measure the final swatch/label pairs with the SVG's real font fallback."""
    root = ET.fromstring(svg)
    legend = list(root)[-8:]
    assert [e.tag.rsplit("}", 1)[-1] for e in legend] == ["rect", "text"] * 4
    for i, element in enumerate(legend):
        element.set("id", f"legend-{i}")
    inkscape = shutil.which("inkscape")
    if not inkscape:
        raise RuntimeError("Inkscape is required for the rendered legend check")
    with tempfile.TemporaryDirectory(prefix="cudapp-legend-") as work:
        path = Path(work) / "stack.svg"
        ET.ElementTree(root).write(path)
        env = dict(os.environ, HOME=work, XDG_CACHE_HOME=work)
        result = subprocess.run([inkscape, str(path), "--query-all"], env=env,
                                check=True, capture_output=True, text=True)
    bounds = {}
    for line in result.stdout.splitlines():
        fields = line.split(",")
        if fields[0].startswith("legend-"):
            bounds[fields[0]] = tuple(map(float, fields[1:]))
    return root, [bounds[f"legend-{i}"] for i in range(len(legend))]


def assert_clear(test, svg):
    root, boxes = legend_bounds(svg)
    width, height = float(root.get("width")), float(root.get("height"))
    for i, (x, y, w, h) in enumerate(boxes):
        test.assertGreaterEqual(x, 0, f"legend item {i} outside left edge")
        test.assertGreaterEqual(y, 0, f"legend item {i} outside top edge")
        test.assertLessEqual(x + w, width, f"legend item {i} clipped at right edge")
        test.assertLessEqual(y + h, height, f"legend item {i} clipped at bottom edge")
        for j, (xx, yy, ww, hh) in enumerate(boxes[:i]):
            overlap = x < xx + ww and xx < x + w and y < yy + hh and yy < y + h
            test.assertFalse(overlap, f"legend items {j} and {i} overlap")


class StackLegendTest(unittest.TestCase):
    def test_rendered_labels_and_swatches_do_not_overlap(self):
        assert_clear(self, fig_stack().svg())

    def test_overlapping_swatch_is_rejected(self):
        root = ET.fromstring(fig_stack().svg())
        legend = list(root)[-8:]
        legend[4].set("x", legend[3].get("x"))
        with self.assertRaisesRegex(AssertionError, "overlap"):
            assert_clear(self, ET.tostring(root, encoding="unicode"))


if __name__ == "__main__":
    unittest.main()
