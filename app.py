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
import requests
import spacy

nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('portuguese'))
nlp = spacy.load("pt_core_news_sm")

# URLs das APIs
SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API = "https://api.crossref.org/works"

# Função para obter artigos mais citados
def get_popular_phrases(query, limit=10):
    suggested_phrases = []

    # Pesquisa na API Semantic Scholar
    semantic_params = {"query": query, "limit": limit // 3, "fields": "title,abstract,url,externalIds"}
    semantic_response = requests.get(SEMANTIC_API, params=semantic_params)

    if semantic_response.status_code == 200:
        semantic_data = semantic_response.json().get("data", [])
        for item in semantic_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', '')}. {item.get('abstract', '')}",
                "doi": item['externalIds'].get('DOI', 'N/A'),
                "link": item.get('url', 'N/A')
            })

    # Pesquisa na API CrossRef
    crossref_params = {"query": query, "rows": limit // 3}
    crossref_response = requests.get(CROSSREF_API, params=crossref_params)

    if crossref_response.status_code == 200:
        crossref_data = crossref_response.json().get("message", {}).get("items", [])
        for item in crossref_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', [''])[0]}. {item.get('abstract', '')}",
                "doi": item.get('DOI', 'N/A'),
                "link": item.get('URL', 'N/A')
            })

    return suggested_phrases

# Função para gerar frases recomendadas com base nas palavras mais comuns
def generate_suggested_sentences(suggested_phrases):
    all_phrases_text = " ".join([item['phrase'] for item in suggested_phrases])
    doc = nlp(all_phrases_text)
    words = [token.text for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
    common_words = Counter(words).most_common(10)

    suggested_sentences = [
        f"Incorporar conceitos como '{word}' pode enriquecer a argumentação."
        for word, freq in common_words
    ]

    return suggested_sentences

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

    content = [
        Paragraph("<b>Relatório de Sugestão de Melhorias no Artigo</b>", styles['Title']),
        Paragraph(f"<b>Tema Identificado com base nas principais palavras do artigo:</b> {tema}", justified_style),
        Paragraph(f"<b>Probabilidade do artigo ser uma referência:</b> {probabilidade}%", justified_style),
        Paragraph(f"<b>Explicação:</b> {descricao}", justified_style)
    ]

    content.append(Paragraph("<b>Artigos mais acessados e/ou citados nos últimos 5 anos:</b>", styles['Heading3']))
    if suggested_phrases:
        for item in suggested_phrases:
            content.append(Paragraph(f"• {item['phrase']}<br/><b>DOI:</b> {item['doi']}<br/><b>Link:</b> {item['link']}", justified_style))

    content.append(Paragraph("<b>Frases recomendadas para enriquecer seu artigo:</b>", styles['Heading3']))
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

        # Buscando artigos e frases populares com base no tema identificado
        suggested_phrases = get_popular_phrases(tema, limit=10)
        suggested_sentences = generate_suggested_sentences(suggested_phrases)

        st.success(f"✅ Tema identificado: {tema}")
        st.write(f"📈 Probabilidade de ser uma referência: {probabilidade}%")
        st.write(f"ℹ️ {descricao}")

        generate_report(suggested_sentences, suggested_phrases, tema, probabilidade, descricao)
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

if __name__ == "__main__":
    main()
