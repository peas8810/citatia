import streamlit as st
import pdfplumber
from collections import Counter
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import numpy as np
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
    latest_year = 0  

    # Pesquisa na API Semantic Scholar
    semantic_params = {"query": query, "limit": limit, "fields": "title,abstract,year,url,externalIds"}
    semantic_response = requests.get(SEMANTIC_API, params=semantic_params)

    total_articles_semantic = 0
    if semantic_response.status_code == 200:
        semantic_data = semantic_response.json().get("data", [])
        total_articles_semantic = len(semantic_data)
        for item in semantic_data:
            year = item.get('year', 0)
            latest_year = max(latest_year, year)
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
            year = item.get('published-print', {}).get('date-parts', [[0]])[0][0]
            latest_year = max(latest_year, year)
            suggested_phrases.append({
                "phrase": f"{item.get('title', [''])[0]}. {item.get('abstract', '')}",
                "doi": item.get('DOI', 'N/A'),
                "link": item.get('URL', 'N/A')
            })

    total_references = len(suggested_phrases)
    total_articles = total_articles_semantic + total_articles_crossref

    return suggested_phrases, total_references, total_articles, latest_year

# Nova função para calcular a probabilidade usando a nova fórmula estatística
def evaluate_article_relevance(total_references, total_articles, latest_year, current_year=2025):
    if total_articles == 0:
        return "0.00% a 0.00%", "Nenhum artigo encontrado para calcular a probabilidade."

    # Probabilidade base
    base_probability = (total_references / total_articles) * 100

    # Fator Temporal
    k = 0.1  # Constante de decaimento
    F_ano = 1 + 1 / (1 + np.exp(-k * (current_year - latest_year)))

    # Probabilidade Final
    final_probability = base_probability * F_ano

    # Ajustando Faixas Percentuais
    faixa_min = max(5, final_probability - 10)
    faixa_max = min(95, final_probability + 10)

    # Descrição baseada na faixa final
    if faixa_max >= 70:
        descricao = "O tema tem alta probabilidade de destaque, com muitas referências relacionadas."
    elif 30 <= faixa_max < 70:
        descricao = "O tema apresenta uma probabilidade moderada de destaque."
    else:
        descricao = "Poucas referências encontradas, o que pode indicar maior chance de originalidade."

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
        suggested_phrases, total_references, total_articles, latest_year = get_popular_phrases(tema, limit=10)

        # Calculando a probabilidade com base nas referências encontradas
        probabilidade, descricao = evaluate_article_relevance(total_references, total_articles, latest_year)

        st.success(f"✅ Tema identificado: {tema}")
        st.write(f"📈 Probabilidade de ser uma referência: {probabilidade}")
        st.write(f"ℹ️ {descricao}")

        # Baixar relatório (opcional)
        with open("report.pdf", "wb") as f:
            f.write("Relatório gerado com sucesso".encode())  # Exemplo de conteúdo fictício
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

if __name__ == "__main__":
    main()
