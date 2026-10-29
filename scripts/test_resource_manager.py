"""Unit tests for SALAZAR's ResourceManager."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.resource_manager import ResourceManager


class TestResourceManager(unittest.TestCase):
    def test_default_threshold(self):
        manager = ResourceManager()
        self.assertEqual(manager.max_ram_used_percent, 90.0)

    def test_custom_threshold(self):
        manager = ResourceManager(max_ram_used_percent=75)
        self.assertEqual(manager.max_ram_used_percent, 75.0)

    def test_invalid_thresholds_rejected(self):
        for threshold in (0, -1, 100.1):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    ResourceManager(threshold)

    def test_snapshot_with_gpu(self):
        gpu = {
            "name": "Test GPU",
            "vram_total_mib": 6144,
            "vram_used_mib": 512,
            "vram_free_mib": 5632,
            "gpu_utilization_percent": 10,
        }
        memory = SimpleNamespace(
            total=16 * 1024**3,
            available=12 * 1024**3,
            percent=25.0,
        )

        with (
            patch("app.resource_manager.psutil.virtual_memory", return_value=memory),
            patch("app.resource_manager.psutil.cpu_count", return_value=12),
            patch("app.resource_manager.psutil.cpu_percent", return_value=5.0),
            patch.object(ResourceManager, "_query_gpu", return_value=gpu),
        ):
            snapshot = ResourceManager().snapshot()

        self.assertEqual(snapshot.cpu_count, 12)
        self.assertEqual(snapshot.cpu_percent, 5.0)
        self.assertEqual(snapshot.ram_total_gib, 16.0)
        self.assertEqual(snapshot.ram_available_gib, 12.0)
        self.assertTrue(snapshot.gpu_available)
        self.assertEqual(snapshot.gpu_name, "Test GPU")
        self.assertEqual(snapshot.vram_free_mib, 5632)

    def test_snapshot_without_gpu(self):
        memory = SimpleNamespace(
            total=8 * 1024**3,
            available=6 * 1024**3,
            percent=25.0,
        )

        with (
            patch("app.resource_manager.psutil.virtual_memory", return_value=memory),
            patch("app.resource_manager.psutil.cpu_count", return_value=4),
            patch("app.resource_manager.psutil.cpu_percent", return_value=2.0),
            patch.object(ResourceManager, "_query_gpu", return_value=None),
        ):
            snapshot = ResourceManager().snapshot()

        self.assertFalse(snapshot.gpu_available)
        self.assertIsNone(snapshot.gpu_name)
        self.assertIsNone(snapshot.vram_total_mib)

    def test_task_allowed_below_threshold(self):
        memory = SimpleNamespace(
            total=16 * 1024**3,
            available=8 * 1024**3,
        )
        with patch(
            "app.resource_manager.psutil.virtual_memory",
            return_value=memory,
        ):
            self.assertTrue(ResourceManager(90).can_start_task(1.0))

    def test_task_rejected_above_threshold(self):
        memory = SimpleNamespace(
            total=16 * 1024**3,
            available=2 * 1024**3,
        )
        with patch(
            "app.resource_manager.psutil.virtual_memory",
            return_value=memory,
        ):
            self.assertFalse(ResourceManager(90).can_start_task(1.0))

    def test_negative_task_estimate_rejected(self):
        with self.assertRaises(ValueError):
            ResourceManager().can_start_task(-0.5)

    def test_zero_memory_estimate(self):
        memory = SimpleNamespace(
            total=16 * 1024**3,
            available=8 * 1024**3,
        )
        with patch(
            "app.resource_manager.psutil.virtual_memory",
            return_value=memory,
        ):
            self.assertTrue(ResourceManager().can_start_task())


if __name__ == "__main__":
    unittest.main(verbosity=2)
