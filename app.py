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


def add_columns(df):
    df = df.copy()
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Revenue"] = pd.to_numeric(df["Quantity"], errors="coerce") * pd.to_numeric(df["UnitPrice"], errors="coerce")
    if "InvoiceDate" in df.columns:
        df["Month"] = pd.to_datetime(df["InvoiceDate"], errors="coerce").dt.strftime("%Y-%m")
    return df


def group_stats(df):
    numeric = df.select_dtypes("number")
    value = "Revenue" if "Revenue" in df.columns else (numeric.columns[0] if len(numeric.columns) else None)
    groups = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col != "Month" and not 2 <= df[col].nunique() <= 40:
            continue
        if value:
            g = df.groupby(col)[value].agg(["sum", "count", "mean"]).round(2)
            g.columns = ["сума", "рядків", "середнє"]
        else:
            g = df.groupby(col).size().to_frame("рядків")
        g = g.sort_index() if col == "Month" else g.sort_values(g.columns[0], ascending=False).head(10)
        groups[col] = g
    return value, groups


def hypotheses_prompt(df, value, groups):
    describe = df.select_dtypes("number").describe().T.round(2).to_string()
    text = "\n\n".join(f"Групування за {col}:\n{g.to_string()}" for col, g in groups.items())
    return f"""Ти аналітик даних. Нижче статистики датасету, які вже пораховані в pandas.
Запропонуй від 5 до 7 гіпотез, які варто перевірити. Пиши українською мовою.

Правила:
- кожна гіпотеза обов'язково спирається на конкретну цифру з даних нижче
- бери числа точно з таблиць, нічого не рахуй сам і не вигадуй
- не повторюй одну й ту саму думку різними словами

Рядків: {len(df)}
Основна числова колонка: {value}

Статистики числових колонок:
{describe}

{text}

Поверни JSON такого вигляду:
{{"hypotheses": [{{"hypothesis": "формулювання гіпотези", "evidence": "цифра з даних, на яку вона спирається", "check": "як її перевірити"}}]}}
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

data = add_columns(df)

with tab_hyp:
    value, groups = group_stats(data)
    if not groups:
        st.write("Не знайшлося текстових колонок, за якими можна згрупувати дані.")
    for col, g in groups.items():
        st.write(f"Групування за {col}" + (f" (колонка {value})" if value else ""))
        left, right = st.columns(2)
        left.dataframe(g)
        if col == "Month":
            right.line_chart(g[g.columns[0]])
        else:
            right.bar_chart(g[g.columns[0]])
    if st.button("Згенерувати гіпотези", key="hyp_button"):
        st.session_state["hyp"] = ask_llm(hypotheses_prompt(data, value, groups))
    result = st.session_state.get("hyp")
    if result:
        st.subheader("Гіпотези від AI")
        for i, h in enumerate(result.get("hypotheses", []), 1):
            st.markdown(f"**{i}. {h.get('hypothesis', '')}**  \nЦифра: {h.get('evidence', '')}  \nЯк перевірити: {h.get('check', '')}")

with tab_sum:
    st.write("Тут буде короткий звіт для керівника.")
