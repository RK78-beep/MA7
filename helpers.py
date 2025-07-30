import pandas as pd
import pdfplumber
from fpdf import FPDF
import tempfile
import random

def parse_excel(file):
    return pd.read_excel(file)

def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return pd.DataFrame({"Extracted_Text": [text]})

def recommend_deal(row):
    return "Proceed" if row["Prediction"] == 1 else "Do Not Proceed"

def gpt_commentary(row):
    if row["Prediction"] == 1:
        return f"Strong fundamentals. Deal looks viable with synergy potential."
    else:
        return f"High risk. Misalignment in valuation or financials could be concern."

def generate_pdf_report(df):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for _, row in df.iterrows():
        line = f"Prediction: {row['Prediction']}, Recommendation: {row['Recommendation']}, Comment: {row['GPT_Commentary']}"
        pdf.multi_cell(0, 10, line)
    pdf.output(temp.name)
    return temp.name
