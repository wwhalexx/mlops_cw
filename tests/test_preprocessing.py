import pytest

from toxic_mlops.preprocessing import TextPreprocessor, clean_text, duplicate_key, preprocess


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<b>Привет</b>\n мир\u200b", "Привет мир"),
        (" НЕ плохая работа! ", "НЕ плохая работа!"),
        ("&amp;lt;b&amp;gt;Текст&amp;lt;/b&amp;gt;", "Текст"),
        ("Ёжик 😃", "Ёжик 😃"),
    ],
)
def test_clean(text, expected):
    assert clean_text(text) == expected
    assert clean_text(clean_text(text)) == expected


def test_versions():
    text = "Ёжик @alex https://example.org/abc НЕ плохой!"
    assert preprocess(text, "v1") == text
    assert preprocess(text, "v2") == "ежик user url не плохой!"
    assert duplicate_key("ЁЖИК") == duplicate_key("ежик")
    transformer = TextPreprocessor("v2")
    assert transformer.fit_transform([text]) == [preprocess(text, "v2")]
    with pytest.raises(ValueError, match="Unknown"):
        preprocess(text, "v3")
