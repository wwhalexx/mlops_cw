"""Integration check: run the real Streamlit page against an already running API."""

import json

from streamlit.testing.v1 import AppTest

app = AppTest.from_file("src/toxic_mlops/ui.py", default_timeout=30).run()
assert not app.exception
app.text_area[0].input("Спасибо за помощь и подробное объяснение!")
app.button[0].click().run()
assert not app.exception and len(app.subheader) == 1
assert any("sha256:" in caption.value for caption in app.caption)
success = {"label": app.subheader[0].value, "version": app.caption[0].value}
app.text_area[0].input("")
app.button[0].click().run()
assert not app.exception and len(app.error) == 1
assert "422" in app.error[0].value
print(
    json.dumps(
        {"successful_prediction": success, "invalid_input": app.error[0].value},
        ensure_ascii=False,
        indent=2,
    )
)
