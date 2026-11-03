import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WebDataCase(unittest.TestCase):
    def test_the_web_app_data_matches_the_python_app(self):
        # web/src/data/*.json must be regenerated (python tools/export_web_data.py) after settings change.
        result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "export_web_data.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
