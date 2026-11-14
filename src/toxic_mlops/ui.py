"""Run with python -m streamlit; all predictions go through the HTTP API."""

import streamlit as st

from toxic_mlops.client import request_prediction, validate_input
from toxic_mlops.config import load_settings

st.set_page_config(page_title="Проверка токсичности", page_icon="💬")
st.title("Проверка токсичности комментария")
st.write("Модель может ошибаться. Результат предназначен для учебной демонстрации.")
settings = load_settings()
st.caption(
    f"От {settings.inference.min_chars} символов после очистки "
    f"до {settings.inference.max_chars} введённых символов. "
    "Лишние пробелы и HTML-теги не учитываются в минимальной длине."
)
with st.form("prediction"):
    text = st.text_area("Комментарий", max_chars=settings.inference.max_chars)
    submitted = st.form_submit_button("Проверить")
if submitted:
    try:
        validate_input(text, settings.inference.min_chars, settings.inference.max_chars)
        result = request_prediction(settings.inference.api_url, text)
        st.subheader("Токсичный" if result["label"] else "Нетоксичный")
        st.write(f"Оценка токсичности: {result['score']:.3f}")
        st.caption(f"Версия модели: {result['model_version']}")
        st.caption(f"Предобработка: {result['preprocessing_version']}")
    except ValueError as error:
        st.error(str(error))
