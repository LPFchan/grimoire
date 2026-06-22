"""Tests for the manager->proxy shared route table."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grimoire.proxy.routes_table import publish, RouteTableReader


class RouteTableTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "routes.json")

    def test_publish_and_read_replicas(self):
        publish({"emb": {"status": "loaded", "replicas": [
            {"port": 8011, "backend_model_id": "emb"},
            {"port": 8001, "backend_model_id": "emb"},
        ]}}, path=self.path)
        r = RouteTableReader(self.path)
        reps = r.replicas("emb")
        self.assertEqual([x["port"] for x in reps], [8011, 8001])

    def test_unknown_model_returns_empty(self):
        publish({"emb": {"status": "loaded", "replicas": [{"port": 8011, "backend_model_id": "emb"}]}}, path=self.path)
        self.assertEqual(RouteTableReader(self.path).replicas("nope"), [])

    def test_unloaded_status_returns_empty(self):
        publish({"emb": {"status": "loading", "replicas": [{"port": 8011, "backend_model_id": "emb"}]}}, path=self.path)
        self.assertEqual(RouteTableReader(self.path).replicas("emb"), [])

    def test_reader_picks_up_changes(self):
        publish({"emb": {"status": "loaded", "replicas": [{"port": 8011, "backend_model_id": "emb"}]}}, path=self.path)
        r = RouteTableReader(self.path)
        self.assertEqual(len(r.replicas("emb")), 1)
        time.sleep(0.01)
        publish({"emb": {"status": "loaded", "replicas": [
            {"port": 8011, "backend_model_id": "emb"}, {"port": 8001, "backend_model_id": "emb"}]}}, path=self.path)
        self.assertEqual(len(r.replicas("emb")), 2)

    def test_missing_file_returns_empty(self):
        self.assertEqual(RouteTableReader(self.path + ".missing").replicas("emb"), [])


if __name__ == "__main__":
    unittest.main()
