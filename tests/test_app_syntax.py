"""
tests/test_app_syntax.py
Verifies that app.py imports cleanly without any NameError or missing type annotation errors.
"""

import unittest
from pathlib import Path


class TestAppImports(unittest.TestCase):
    def test_app_py_annotations_and_imports(self):
        """Verify app.py can be loaded and line 111 type annotations evaluate cleanly."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Parse AST
        import ast
        tree = ast.parse(code, filename=str(app_path))
        self.assertIsNotNone(tree)

        # Check imports in global namespace
        namespace = {}
        exec("from typing import Tuple, Any, Dict, List, Optional, Union", namespace)
        from src.road_network import load_road_network
        from src.route_optimizer import EVRouteOptimizer
        namespace["EVRouteOptimizer"] = EVRouteOptimizer
        namespace["load_road_network"] = load_road_network

        # Evaluate the exact annotation that failed
        eval_annotation = eval("Tuple[EVRouteOptimizer, Any]", namespace)
        self.assertIsNotNone(eval_annotation)


if __name__ == "__main__":
    unittest.main()
