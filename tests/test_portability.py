import unittest
from unittest.mock import MagicMock, patch

from app.detector import PersonDetector
from app.main import parse_args
from app.camera import USBCamera


class PortabilityTests(unittest.TestCase):
    def session(self, available, provider="auto", fail=False):
        fake = MagicMock()
        fake.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 640, 640])]
        with patch("app.detector.ort.get_available_providers", return_value=available), patch(
            "app.detector.ort.InferenceSession",
            side_effect=[RuntimeError("GPU failed"), fake] if fail else None,
            return_value=fake,
        ) as create:
            PersonDetector(provider=provider)
            return create.call_args_list

    def test_cpu_only(self):
        calls = self.session(["CPUExecutionProvider"])
        self.assertEqual(calls[0].kwargs["providers"], ["CPUExecutionProvider"])

    def test_gpu_auto(self):
        calls = self.session(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(calls[0].kwargs["providers"], ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_gpu_failure_falls_back(self):
        calls = self.session(["DmlExecutionProvider", "CPUExecutionProvider"], fail=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1].kwargs["providers"], ["CPUExecutionProvider"])

    def test_explicit_provider_unavailable(self):
        with self.assertRaises(ValueError):
            self.session(["CPUExecutionProvider"], "cuda")

    def test_cli(self):
        args = parse_args(["--video", "other.mp4", "--provider", "cpu", "--display-width", "800"])
        self.assertEqual(str(args.video), "other.mp4")
        self.assertEqual(args.provider, "cpu")
        self.assertEqual(args.display_width, 800)

    def test_invalid_options(self):
        with self.assertRaises(SystemExit):
            parse_args(["--confidence", "2"])
        with self.assertRaises(SystemExit):
            parse_args(["--display-width", "0"])

    def test_camera_uses_portable_default(self):
        self.assertEqual(USBCamera().backend, "AUTO")


if __name__ == "__main__":
    unittest.main()
