import pandas as pd

df = pd.read_csv("data/data.csv", encoding="ISO-8859-1", dtype={"InvoiceNo": str, "CustomerID": str})

invoices = df["InvoiceNo"].drop_duplicates().sample(frac=0.033, random_state=42)
sample = df[df["InvoiceNo"].isin(invoices)]

sample.to_csv("data/sample.csv", index=False)
print("Всього рядків:", len(df))
print("Рядків у sample.csv:", len(sample))
print("Замовлень у sample.csv:", sample["InvoiceNo"].nunique())
