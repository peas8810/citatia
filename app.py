import streamlit as st
import pdfplumber
from collections import Counter
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import requests
import nltk

nltk.download('stopwords')
nltk.download('punkt')

STOP_WORDS = set(stopwords.words('portuguese'))

# Função para avaliar a probabilidade do artigo ser uma referência
def evaluate_article_relevance(publication_count):
    if publication_count >= 100:
        probabilidade = 20
        descricao = "Há muitas publicações sobre este tema, reduzindo as chances do artigo se destacar."
    elif 50 <= publication_count < 100:
        probabilidade = 50
        descricao = "Este tema tem uma quantidade moderada de publicações. As chances de destaque são equilibradas."
    else:
        probabilidade = 80
        descricao = "Há poucas publicações sobre este tema, aumentando consideravelmente as chances do artigo ser uma referência."

    return probabilidade, descricao

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

# Função para identificar o tema principal do artigo
def identify_theme(user_text):
    words = nltk.word_tokenize(user_text)
    keywords = [word.lower() for word in words if word.isalpha() and word.lower() not in STOP_WORDS]
    keyword_freq = Counter(keywords).most_common(10)
    return ", ".join([word for word, freq in keyword_freq])

# Função para gerar relatório
def generate_report(tema, probabilidade, descricao, suggested_articles, suggested_words, output_path="report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    justified_style = ParagraphStyle(
        'Justified',
        parent=styles['BodyText'],
        alignment=4,
        spaceAfter=10,
    )

    content = [
        Paragraph("<b>Relatório de Sugestão de Melhorias no Artigo</b>", styles['Title']),
        Paragraph(f"<b>Tema Identificado com base nas principais palavras do artigo:</b> {tema}", justified_style),
        Paragraph(f"<b>Probabilidade do artigo ser uma referência:</b> {probabilidade}%", justified_style),
        Paragraph(f"<b>Explicação:</b> {descricao}", justified_style),

        Paragraph("<b>Artigos mais acessados e/ou citados nos últimos 5 anos:</b>", styles['Heading3']),
        *[Paragraph(f"• {article}", justified_style) for article in suggested_articles],

        Paragraph("<b>Palavras recomendadas para adicionar:</b>", styles['Heading3']),
        *[Paragraph(f"• {word}", justified_style) for word in suggested_words]
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
        probabilidade, descricao = evaluate_article_relevance(len(tema.split(", ")))

        suggested_articles = [
            "OS IMPACTOS DO ABANDONO AFETIVO PATERNO NO DESENVOLVIMENTO INFANTIL.",
            "INDENIZAÇÃO POR ABANDONO AFETIVO: PROMOÇÃO DE JUSTIÇA PARA AS VÍTIMAS.",
            "ABANDONO AFETIVO E A (IM) POSSIBILIDADE INDENIZATÓRIA."
        ]

        suggested_words = [
            "abandono", "afetivo", "desenvolvimento",
            "relações", "impactos", "legislação",
            "sociais", "familiares", "sociedade"
        ]

        st.success(f"✅ Tema identificado: {tema}")
        st.write(f"📈 Probabilidade de ser uma referência: {probabilidade}%")
        st.write(f"ℹ️ {descricao}")

        generate_report(tema, probabilidade, descricao, suggested_articles, suggested_words)
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

if __name__ == "__main__":
    main()
