import pytest

from infrastructure import datachat_export_store


@pytest.fixture(autouse=True)
def isolated_datachat_exports(tmp_path, monkeypatch):
    """Keep DataChat CSV exports in a per-test directory and registry."""
    export_dir = tmp_path / "datachat_exports"
    monkeypatch.setattr(datachat_export_store, "_export_dir", str(export_dir))
    monkeypatch.setattr(datachat_export_store, "_exports", {})
    return export_dir
