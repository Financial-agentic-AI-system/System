import importlib
from pathlib import Path


def test_project_structure_is_present():
    root = Path(__file__).resolve().parents[2]

    expected_paths = [
        root / "pyproject.toml",
        root / "src",
        root / "tests",
    ]

    for path in expected_paths:
        assert path.exists(), f"Brak oczekiwanego elementu: {path}"


def test_app_package_can_be_imported():
    module = importlib.import_module("src")
    assert module is not None
