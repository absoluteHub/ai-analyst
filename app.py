import pandas as pd
import streamlit as st


def load_csv(file):
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="ISO-8859-1")
    return df


st.set_page_config(page_title="AI-асистент аналітика", layout="wide")
st.title("AI-асистент аналітика")

st.sidebar.header("Дані")
uploaded = st.sidebar.file_uploader("Завантаж CSV-файл", type="csv")
if st.sidebar.button("Використати демо-датасет"):
    st.session_state["demo"] = True

if uploaded is not None:
    source = uploaded
elif st.session_state.get("demo"):
    source = "data/sample.csv"
else:
    st.info("Завантаж CSV у меню зліва або натисни кнопку з демо-датасетом.")
    st.stop()

try:
    df = load_csv(source)
except pd.errors.EmptyDataError:
    st.error("Файл порожній. Завантаж CSV з даними.")
    st.stop()
except Exception:
    st.error("Не вдалося прочитати файл як CSV. Перевір, що це справді CSV з комами між колонками.")
    st.stop()

if df.empty:
    st.error("У файлі є тільки заголовки, а рядків з даними немає.")
    st.stop()

st.subheader("Перші рядки")
st.write(f"Рядків: {df.shape[0]}, колонок: {df.shape[1]}")
st.dataframe(df.head(10))

tab_eda, tab_hyp, tab_sum = st.tabs(["EDA", "Гіпотези", "Executive summary"])

with tab_eda:
    st.write("Тут буде автоматичний аналіз якості даних.")

with tab_hyp:
    st.write("Тут будуть гіпотези.")

with tab_sum:
    st.write("Тут буде короткий звіт для керівника.")
