import streamlit as st
import pdfplumber
from collections import Counter
from nltk.corpus import stopwords
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import requests
import os
import nltk
import spacy.cli

spacy.cli.download("pt_core_news_sm")
nlp = spacy.load("pt_core_news_sm")
nltk.download('stopwords')
spacy.cli.download("pt_core_news_sm")
nlp = spacy.load("pt_core_news_sm")

STOP_WORDS = set(stopwords.words('portuguese'))

# Modelo PyTorch para prever chance de ser referência
class ArticlePredictor(nn.Module):
    def __init__(self):
        super(ArticlePredictor, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

# Função para identificar o tema principal do artigo
def identify_theme(user_text):
    doc = nlp(user_text)
    keywords = [token.text for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
    keyword_freq = Counter(keywords).most_common(10)
    return ", ".join([word for word, freq in keyword_freq])

# Função para gerar relatório
def generate_report(tema, probabilidade, descricao, output_path="report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    justified_style = ParagraphStyle(
        'Justified',
        parent=styles['BodyText'],
        alignment=4,
        spaceAfter=10,
    )

    content = [
        Paragraph(f"<b>Tema Identificado:</b> {tema}", justified_style),
        Paragraph(f"<b>Probabilidade do artigo ser uma referência:</b> {probabilidade}%", justified_style),
        Paragraph(f"<b>Explicação:</b> {descricao}", justified_style)
    ]

    doc.build(content)

# Interface com Streamlit
def main():
    st.title("Citatia - Analisador de Artigos Acadêmicos")
    st.write("Faça o upload do seu arquivo PDF para iniciar a análise.")

    uploaded_file = st.file_uploader("Envie o arquivo PDF", type='pdf')

    if uploaded_file:
        with open("uploaded_article.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("🔍 Analisando o arquivo...")

        user_text = extract_text_from_pdf("uploaded_article.pdf")
        tema = identify_theme(user_text)
        probabilidade = 75  # Exemplo fixo para simplificação
        descricao = "Tema com potencial de destaque na literatura acadêmica."

        st.success(f"✅ Tema identificado: {tema}")
        st.write(f"📈 Probabilidade de ser uma referência: {probabilidade}%")
        st.write(f"ℹ️ {descricao}")

        generate_report(tema, probabilidade, descricao)
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

if __name__ == "__main__":
    main()
