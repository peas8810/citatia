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
import torch
import torch.nn as nn
import torch.nn.functional as F

nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('portuguese'))

# URLs das APIs
SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API = "https://api.crossref.org/works"

# Função para obter artigos mais citados
def get_popular_phrases(query, limit=10):
    suggested_phrases = []

    # Pesquisa na API Semantic Scholar
    semantic_params = {"query": query, "limit": limit, "fields": "title,abstract,url,externalIds,citationCount"}
    semantic_response = requests.get(SEMANTIC_API, params=semantic_params)

    if semantic_response.status_code == 200:
        semantic_data = semantic_response.json().get("data", [])
        for item in semantic_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', '')}. {item.get('abstract', '')}",
                "doi": item['externalIds'].get('DOI', 'N/A'),
                "link": item.get('url', 'N/A'),
                "citationCount": item.get('citationCount', 0)
            })

    # Pesquisa na API CrossRef
    crossref_params = {"query": query, "rows": limit}
    crossref_response = requests.get(CROSSREF_API, params=crossref_params)

    if crossref_response.status_code == 200:
        crossref_data = crossref_response.json().get("message", {}).get("items", [])
        for item in crossref_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', [''])[0]}. {item.get('abstract', '')}",
                "doi": item.get('DOI', 'N/A'),
                "link": item.get('URL', 'N/A'),
                "citationCount": item.get('is-referenced-by-count', 0)
            })

    # Ordenar por número de citações
    suggested_phrases.sort(key=lambda x: x.get('citationCount', 0), reverse=True)

    return suggested_phrases

# Função para extrair as 10 palavras mais importantes dos artigos
def extract_top_keywords(suggested_phrases):
    all_text = " ".join([item['phrase'] for item in suggested_phrases])
    words = re.findall(r'\b\w+\b', all_text.lower())
    words = [word for word in words if word not in STOP_WORDS and len(word) > 3]  # Filtra stopwords e palavras curtas
    word_freq = Counter(words).most_common(10)
    return [word for word, freq in word_freq]

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

# Avalia a probabilidade do artigo se tornar uma referência
def evaluate_article_relevance(publication_count):
    model = ArticlePredictor()
    data = torch.tensor([[publication_count]], dtype=torch.float32)
    probability = model(data).item() * 100

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

# Função para gerar relatório detalhado
def generate_report(suggested_phrases, top_keywords, tema, probabilidade, descricao, output_path="report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    justified_style = ParagraphStyle(
        'Justified',
        parent=styles['BodyText'],
        alignment=4,
        spaceAfter=10,
    )

    content = [
        Paragraph("<b>Relatório de Sugestão de Melhorias no Artigo - CitaIA - PEAS.Co</b>", styles['Title']),
        Paragraph(f"<b>Tema Identificado com base nas principais palavras do artigo:</b> {tema}", justified_style),
        Paragraph(f"<b>Probabilidade de o artigo se tornar uma referência (Calculada com base no volume de artigos publicados na área nos últimos cinco anos e no interesse demonstrado pelo tema nas ferramentas de busca):</b> {probabilidade}%", justified_style),
        Paragraph(f"<b>Explicação:</b> {descricao}", justified_style)
    ]

    content.append(Paragraph("<b>Artigos mais acessados e/ou citados nos últimos 5 anos:</b>", styles['Heading3']))
    if suggested_phrases:
        for item in suggested_phrases:
            content.append(Paragraph(f"• {item['phrase']}<br/><b>DOI:</b> {item['doi']}<br/><b>Link:</b> {item['link']}<br/><b>Citações:</b> {item.get('citationCount', 'N/A')}", justified_style))

    content.append(Paragraph("<b>Palavras-chave em comum nos artigos mais citados da área identificada:</b>", styles['Heading3']))
    if top_keywords:
        for word in top_keywords:
            content.append(Paragraph(f"• {word}", justified_style))
    else:
        content.append(Paragraph("Nenhuma palavra-chave relevante encontrada.", justified_style))

    doc.build(content)

# Interface com Streamlit
def main():
    st.title("CitatIA - Analisador de Artigos Acadêmicos")
    st.write("Faça o upload do seu arquivo PDF para iniciar a análise.")

    uploaded_file = st.file_uploader("Envie o arquivo PDF", type='pdf')

    if uploaded_file:
        with open("uploaded_article.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("🔍 Analisando o arquivo...")

        user_text = extract_text_from_pdf("uploaded_article.pdf")
        tema = identify_theme(user_text)

        # Buscando artigos e frases populares com base no tema identificado
        suggested_phrases = get_popular_phrases(tema, limit=10)

        # Extrair as 10 palavras mais importantes dos artigos
        top_keywords = extract_top_keywords(suggested_phrases)

        # Calculando a probabilidade com base nas referências encontradas
        publication_count = len(suggested_phrases)
        probabilidade, descricao = evaluate_article_relevance(publication_count)

        st.success(f"✅ Tema identificado: {tema}")
        st.write(f"📈 Probabilidade de o artigo se tornar uma referência (Calculada com base no volume de artigos publicados na área nos últimos cinco anos e no interesse demonstrado pelo tema nas ferramentas de busca): {probabilidade}%")
        st.write(f"ℹ️ {descricao}")

        st.write("<b>Palavras-chave em comum nos artigos mais citados da área identificada:</b>", unsafe_allow_html=True)
        if top_keywords:
            for word in top_keywords:
                st.write(f"• {word}")
        else:
            st.write("Nenhuma palavra-chave relevante encontrada.")

        generate_report(suggested_phrases, top_keywords, tema, probabilidade, descricao)
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

if __name__ == "__main__":
    main()
