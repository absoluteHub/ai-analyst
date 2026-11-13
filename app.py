import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()


def load_csv(file):
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="ISO-8859-1")
    return df


def ask_llm(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Немає API-ключа. Створи файл .env за зразком .env.example і впиши туди GEMINI_API_KEY.")
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
    except Exception as e:
        st.error(f"Помилка Gemini API: {e}")
        return None
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        st.error("Модель повернула невалідний JSON. Спробуй натиснути кнопку ще раз.")
        return None


def eda_stats(df):
    numeric = df.select_dtypes("number")
    table = pd.DataFrame({
        "тип": df.dtypes.astype(str),
        "пропуски": df.isna().sum(),
        "пропуски, %": (df.isna().mean() * 100).round(1),
        "унікальних": df.nunique(),
    })
    table["від'ємних"] = (numeric < 0).sum().reindex(df.columns).astype("Int64")
    table["нульових"] = (numeric == 0).sum().reindex(df.columns).astype("Int64")
    describe = numeric.describe().T.round(2) if not numeric.empty else pd.DataFrame()
    duplicates = int(df.duplicated().sum())
    return table, describe, duplicates


def eda_prompt(df, table, describe, duplicates):
    return f"""Ти аналітик даних. Тобі передали статистики якості датасету, які вже пораховані в pandas.
Напиши висновки про якість даних українською мовою.

Правила:
- використовуй тільки числа з даних нижче, нічого не рахуй сам і не вигадуй
- якщо згадуєш число, бери його точно з таблиць
- пиши коротко і по суті

Рядків: {df.shape[0]}
Колонок: {df.shape[1]}
Повних дублікатів рядків: {duplicates}

Колонки (тип, пропуски, унікальні, від'ємні і нульові значення):
{table.to_string()}

Статистики числових колонок:
{describe.to_string()}

Поверни JSON такого вигляду:
{{"summary": "2-3 речення загального висновку",
 "problems": [{{"column": "назва колонки", "problem": "що не так, з цифрою", "recommendation": "що з цим зробити"}}]}}
"""


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
    table, describe, duplicates = eda_stats(df)
    st.write(f"Повних дублікатів рядків: {duplicates}")
    st.dataframe(table)
    if not describe.empty:
        st.write("Статистики числових колонок")
        st.dataframe(describe)
    if st.button("Отримати висновки від AI", key="eda_button"):
        st.session_state["eda"] = ask_llm(eda_prompt(df, table, describe, duplicates))
    result = st.session_state.get("eda")
    if result:
        st.subheader("Висновки AI")
        st.write(result.get("summary", ""))
        for p in result.get("problems", []):
            st.markdown(f"- **{p.get('column', '')}**: {p.get('problem', '')} Рекомендація: {p.get('recommendation', '')}")

with tab_hyp:
    st.write("Тут будуть гіпотези.")

with tab_sum:
    st.write("Тут буде короткий звіт для керівника.")
