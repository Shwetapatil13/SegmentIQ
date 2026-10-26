import os
import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_smoke():
    """Verify that app.py loads without raising any runtime exceptions."""
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"App crashed on startup with exception: {at.exception}"
