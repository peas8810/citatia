import streamlit as st
import pdfplumber
from collections import Counter
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import nltk
import re
import requests

nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('portuguese'))

# URLs das APIs
SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API = "https://api.crossref.org/works"

# Função para obter artigos mais citados
def get_popular_phrases(query, limit=10):
    suggested_phrases = []

    # Pesquisa na API Semantic Scholar
    semantic_params = {"query": query, "limit": limit, "fields": "title,abstract,url,externalIds"}
    semantic_response = requests.get(SEMANTIC_API, params=semantic_params)

    total_articles_semantic = 0
    if semantic_response.status_code == 200:
        semantic_data = semantic_response.json().get("data", [])
        total_articles_semantic = len(semantic_data)
        for item in semantic_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', '')}. {item.get('abstract', '')}",
                "doi": item['externalIds'].get('DOI', 'N/A'),
                "link": item.get('url', 'N/A')
            })

    # Pesquisa na API CrossRef
    crossref_params = {"query": query, "rows": limit}
    crossref_response = requests.get(CROSSREF_API, params=crossref_params)

    total_articles_crossref = 0
    if crossref_response.status_code == 200:
        crossref_data = crossref_response.json().get("message", {}).get("items", [])
        total_articles_crossref = len(crossref_data)
        for item in crossref_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', [''])[0]}. {item.get('abstract', '')}",
                "doi": item.get('DOI', 'N/A'),
                "link": item.get('URL', 'N/A')
            })

    total_references = len(suggested_phrases)
    total_articles = total_articles_semantic + total_articles_crossref

    return suggested_phrases, total_references, total_articles

# Função para calcular a probabilidade com faixa percentual
def evaluate_article_relevance(total_references, total_articles):
    if total_articles == 0:
        return 0.0, "Nenhum artigo encontrado para calcular a probabilidade."

    # Cálculo da probabilidade com faixas mais realistas
    probability = (total_references / total_articles) * 100
    faixa_min = max(5, probability - 10)
    faixa_max = min(95, probability + 10)

    descricao = "O tema apresenta uma probabilidade moderada de destaque." if 30 <= probability < 70 else \
                "Poucas referências encontradas, o que pode indicar maior chance de originalidade." if probability < 30 else \
                "O tema tem alta probabilidade de destaque, com muitas referências relacionadas."

    probabilidade_faixa = f"{faixa_min:.2f}% a {faixa_max:.2f}%"
    return probabilidade_faixa, descricao

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

# Função para gerar relatório detalhado
def generate_report(suggested_phrases, tema, probabilidade, descricao, output_path="report.pdf"):
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
        Paragraph(f"<b>Probabilidade do artigo ser uma referência:</b> {probabilidade}", justified_style),
        Paragraph(f"<b>Explicação:</b> {descricao}", justified_style)
    ]

    content.append(Paragraph("<b>Artigos mais acessados e/ou citados nos últimos 5 anos:</b>", styles['Heading3']))
    if suggested_phrases:
        for item in suggested_phrases:
            content.append(Paragraph(f"• {item['phrase']}<br/><b>DOI:</b> {item['doi']}<br/><b>Link:</b> {item['link']}", justified_style))

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

        # Buscando artigos e frases populares com base no tema identificado
        suggested_phrases, total_references, total_articles = get_popular_phrases(tema, limit=10)

        # Calculando a probabilidade com base nas referências encontradas
        probabilidade, descricao = evaluate_article_relevance(total_references, total_articles)

        st.success(f"✅ Tema identificado: {tema}")
        st.write(f"📈 Probabilidade de ser uma referência: {probabilidade}")
        st.write(f"ℹ️ {descricao}")

        generate_report(suggested_phrases, tema, probabilidade, descricao)
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

if __name__ == "__main__":
    main()
