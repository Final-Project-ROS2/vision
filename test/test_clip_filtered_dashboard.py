#!/usr/bin/env python3
"""
Unit tests: verify classify_filtered_history.json path, format, and HTTP endpoint.

Run directly (no ROS needed):
    python3 test/test_clip_filtered_dashboard.py
"""

import json
import unittest
import urllib.request
import urllib.error
from pathlib import Path

# ── Path constants ────────────────────────────────────────────────────────────
INSTALL_VISION = Path(
    '/home/group11/final_project_ws/install/vision'
    '/lib/python3.12/site-packages/vision'
)
INSTALL_PKG    = INSTALL_VISION.parent          # site-packages/
SRC_VISION     = Path(
    '/home/group11/final_project_ws/src/vision/vision'
)
SRC_PKG        = SRC_VISION.parent              # src/vision/

DASHBOARD_URL  = 'http://localhost:8080'


def _resolve_package_path(py_file: Path) -> Path:
    """Replicate Path(__file__).parent.parent used in the nodes."""
    return py_file.parent.parent


class TestPathConsistency(unittest.TestCase):
    """Writer and reader must resolve to the same JSON file."""

    def test_install_clip_and_dashboard_same_path(self):
        clip_path = _resolve_package_path(INSTALL_VISION / 'clip_classifier.py') \
                    / 'classify_filtered_history.json'
        dash_path = _resolve_package_path(INSTALL_VISION / 'benchmark_dashboard.py') \
                    / 'classify_filtered_history.json'
        self.assertEqual(clip_path, dash_path,
            f"clip writes to {clip_path} but dashboard reads from {dash_path}")

    def test_src_and_install_expected_dir(self):
        """Install package_path == site-packages, NOT src/vision."""
        install_pkg_path = _resolve_package_path(INSTALL_VISION / 'clip_classifier.py')
        self.assertIn('site-packages', str(install_pkg_path),
            f"install path looks wrong: {install_pkg_path}")

    def test_filtered_file_exists(self):
        fpath = INSTALL_PKG / 'classify_filtered_history.json'
        self.assertTrue(fpath.exists(),
            f"classify_filtered_history.json not found at {fpath}")


class TestFilteredFileFormat(unittest.TestCase):
    """Each record must have the fields the dashboard expects."""

    @classmethod
    def setUpClass(cls):
        fpath = INSTALL_PKG / 'classify_filtered_history.json'
        if fpath.exists():
            with open(fpath) as f:
                cls.records = json.load(f)
        else:
            cls.records = []

    def test_file_not_empty(self):
        self.assertGreater(len(self.records), 0,
            "classify_filtered_history.json is empty – call "
            "/vision/classify_bbox_filtered first")

    def test_required_fields_present(self):
        required = {'test_id', 'timestamp', 'label', 'confidence', 'top1_accuracy', 'bbox'}
        for i, rec in enumerate(self.records[:5]):
            missing = required - rec.keys()
            self.assertFalse(missing,
                f"Record #{i} missing fields: {missing}\nRecord: {rec}")

    def test_confidence_is_float_in_range(self):
        for i, rec in enumerate(self.records[:20]):
            c = rec.get('confidence')
            self.assertIsInstance(c, float,
                f"Record #{i}: confidence should be float, got {type(c)}")
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_top1_accuracy_is_null_or_bool(self):
        for i, rec in enumerate(self.records):
            acc = rec.get('top1_accuracy')
            self.assertIn(type(acc), (type(None), bool),
                f"Record #{i}: top1_accuracy should be None or bool, got {type(acc)}: {acc}")

    def test_bbox_is_list_or_dict(self):
        for i, rec in enumerate(self.records[:10]):
            bbox = rec.get('bbox')
            self.assertIsNotNone(bbox, f"Record #{i}: bbox is None")
            self.assertIn(type(bbox), (list, dict),
                f"Record #{i}: bbox type {type(bbox)}")

    def test_no_auto_true_verdicts_in_new_records(self):
        """Records saved after the human-in-the-loop fix should have top1_accuracy=None."""
        # Find the most recent record – should be null, not auto-True
        if not self.records:
            self.skipTest("No records to check")
        last = self.records[-1]
        self.assertIsNone(last.get('top1_accuracy'),
            f"Latest record has auto-verdict instead of null: {last}")


class TestHTTPEndpoint(unittest.TestCase):
    """Verify the dashboard HTTP server exposes /api/classify-filtered-history."""

    def _get(self, path):
        try:
            with urllib.request.urlopen(DASHBOARD_URL + path, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception as e:
            self.skipTest(f"Dashboard not reachable ({e})")

    def test_endpoint_exists_returns_200(self):
        status, data = self._get('/api/classify-filtered-history')
        self.assertEqual(status, 200,
            f"/api/classify-filtered-history returned {status} – endpoint missing or not deployed")

    def test_endpoint_returns_list(self):
        status, data = self._get('/api/classify-filtered-history')
        if status != 200:
            self.skipTest(f"Endpoint returned {status}")
        self.assertIsInstance(data, list,
            f"Expected list, got {type(data)}")

    def test_endpoint_data_matches_file(self):
        fpath = INSTALL_PKG / 'classify_filtered_history.json'
        if not fpath.exists():
            self.skipTest("classify_filtered_history.json not found")
        with open(fpath) as f:
            file_records = json.load(f)

        status, http_records = self._get('/api/classify-filtered-history')
        if status != 200:
            self.skipTest(f"Endpoint returned {status}")

        self.assertEqual(len(file_records), len(http_records),
            f"File has {len(file_records)} records but endpoint returned {len(http_records)}")

    def test_api_data_merges_filtered(self):
        """fetchData() merges clip_classifications + filtered; verify /api/data exists."""
        status, data = self._get('/api/data')
        if status != 200:
            self.skipTest("/api/data not reachable")
        self.assertIn('clip_classifications', data,
            "clip_classifications key missing from /api/data")

    def test_clip_verdict_endpoint_accessible(self):
        """POST /api/clip-verdict must return 4xx for bad payload, not 404."""
        try:
            req = urllib.request.Request(
                DASHBOARD_URL + '/api/clip-verdict',
                data=b'{}',
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception as e:
            self.skipTest(f"Dashboard not reachable ({e})")

        self.assertNotEqual(status, 404,
            "/api/clip-verdict returned 404 – endpoint not deployed in running process")
        self.assertIn(status, (200, 400),
            f"Unexpected status {status} for /api/clip-verdict")


if __name__ == '__main__':
    print("=" * 60)
    print("CLIP Filtered Dashboard – Path & Endpoint Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
