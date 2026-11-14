"""Run: streamlit run src/toxic_mlops/ui.py. All predictions go through HTTP."""

import streamlit as st

from toxic_mlops.client import request_prediction
from toxic_mlops.config import load_settings

st.set_page_config(page_title="Проверка токсичности", page_icon="💬")
st.title("Проверка токсичности комментария")
st.write("Модель может ошибаться. Результат предназначен для учебной демонстрации.")
settings = load_settings()
with st.form("prediction"):
    text = st.text_area("Комментарий", max_chars=settings.inference.max_chars)
    submitted = st.form_submit_button("Проверить")
if submitted:
    try:
        result = request_prediction(settings.inference.api_url, text)
        st.subheader("Токсичный" if result["label"] else "Нетоксичный")
        st.write(f"Оценка токсичности: {result['score']:.3f}")
        st.caption(f"Версия модели: {result['model_version']}")
        st.caption(f"Предобработка: {result['preprocessing_version']}")
    except ValueError as error:
        st.error(str(error))
