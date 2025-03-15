import streamlit as st
import pdfplumber
from collections import Counter
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import nltk
import re

nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('portuguese'))

# Modelo Scikit-Learn para prever chance de ser referência
def evaluate_article_relevance(publication_count):
    model = LogisticRegression()
    X = [[10], [50], [100]]
    y = [1, 0, 0]
    model.fit(X, y)

    probability = model.predict_proba([[publication_count]])[0][1] * 100

    if publication_count >= 100:
        descricao = "Há muitas publicações sobre este tema, reduzindo as chances do artigo se destacar."
    elif 50 <= publication_count < 100:
        descricao = "Este tema tem uma quantidade moderada de publicações. As chances de destaque são equilibradas."
    else:
        descricao = "Há poucas publicações sobre este tema, aumentando consideravelmente as chances do artigo ser uma referência."

    return round(probability, 2), descricao

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

# Função para identificar o tema principal do artigo
def identify_theme(user_text):
    words = re.findall(r'\b\w+\b', user_text)
    keywords = [word.lower() for word in words if word.lower() not in STOP_WORDS]
    keyword_freq = Counter(keywords).most_common(10)
    return ", ".join([word for word, freq in keyword_freq])

# Função para gerar relatório detalhado com artigos, frases e sugestões
def generate_report(suggested_sentences, suggested_phrases, tema, probabilidade, descricao, output_path="report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    justified_style = ParagraphStyle(
        'Justified',
        parent=styles['BodyText'],
        alignment=4,
        spaceAfter=10,
    )

    content = []
    content.append(Paragraph("<b>Relatório de Sugestão de Melhorias no Artigo</b>", styles['Title']))
    content.append(Paragraph(f"<b>Tema Identificado com base nas principais palavras do artigo:</b> {tema}", justified_style))
    content.append(Paragraph(f"<b>Probabilidade do artigo ser uma referência:</b> {probabilidade}%", justified_style))
    content.append(Paragraph(f"<b>Explicação:</b> {descricao}", justified_style))

    content.append(Paragraph("<b>Artigos mais acessados e/ou citados nos últimos 5 anos:</b>", styles['Heading3']))
    if suggested_phrases:
        for item in suggested_phrases:
            content.append(Paragraph(f"• {item['phrase']}<br/><b>DOI:</b> {item['doi']}<br/><b>Link:</b> {item['link']}", justified_style))

    content.append(Paragraph("<b>Palavras recomendadas para adicionar:</b>", styles['Heading3']))
    if suggested_sentences:
        for sentence in suggested_sentences:
            content.append(Paragraph(f"• {sentence}", justified_style))
    else:
        content.append(Paragraph("Nenhuma sugestão de frase relevante encontrada.", justified_style))

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

        # Exemplos fictícios para teste da nova funcionalidade
        suggested_phrases = [
            {"phrase": "Impactos ambientais em áreas urbanas", "doi": "10.1234/abcd123", "link": "https://doi.org/10.1234/abcd123"},
            {"phrase": "Sustentabilidade na construção civil", "doi": "10.5678/wxyz456", "link": "https://doi.org/10.5678/wxyz456"}
        ]

        suggested_sentences = [
            "Sistemas ecológicos sustentáveis",
            "Técnicas avançadas de aproveitamento energético",
            "Estudos recentes sobre gestão hídrica"
        ]

        st.success(f"✅ Tema identificado: {tema}")
        st.write(f"📈 Probabilidade de ser uma referência: {probabilidade}%")
        st.write(f"ℹ️ {descricao}")

        generate_report(suggested_sentences, suggested_phrases, tema, probabilidade, descricao)
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

if __name__ == "__main__":
    main()
