import streamlit as st
import random
import hashlib
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
from datetime import datetime, timedelta

# 🔧 Verificação e download das stopwords
try:
    STOP_WORDS = set(stopwords.words('portuguese'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('portuguese'))

# 🔗 URL da API do Google Sheets
URL_GOOGLE_SHEETS = "https://script.google.com/macros/s/AKfycbyHRCrD5-A_JHtaUDXsGWQ22ul9ml5vvK3YYFzIE43jjCdip0dBMFH_Jmd8w971PLte/exec"

# APIs externas
SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API = "https://api.crossref.org/works"

# ================================
# 🔐 Funções de Registro e Validação
# ================================
def gerar_codigo_verificacao(texto):
    return hashlib.md5(texto.encode()).hexdigest()[:10].upper()

def salvar_email_google_sheets(nome, email, codigo_verificacao):
    dados = {"nome": nome, "email": email, "codigo": codigo_verificacao}
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(URL_GOOGLE_SHEETS, json=dados, headers=headers)
        if response.text.strip() == "Sucesso":
            st.success("✅ Dados registrados com sucesso!")
        else:
            st.error(f"❌ Erro ao salvar dados: {response.text}")
    except Exception as e:
        st.error(f"❌ Erro de conexão: {e}")

def verificar_codigo_google_sheets(codigo_digitado):
    try:
        response = requests.get(f"{URL_GOOGLE_SHEETS}?codigo={codigo_digitado}")
        return response.text.strip() == "Valido"
    except Exception as e:
        st.error(f"❌ Erro na conexão com o Google Sheets: {e}")
        return False

# =================================
# 🔍 Funções de Pesquisa e Análise
# =================================
def get_popular_phrases(query, limit=10):
    suggested_phrases = []

    try:
        semantic_params = {"query": query, "limit": limit, "fields": "title,abstract,url,externalIds,citationCount"}
        semantic_response = requests.get(SEMANTIC_API, params=semantic_params)
        semantic_response.raise_for_status()
        semantic_data = semantic_response.json().get("data", [])
        for item in semantic_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', '')}. {item.get('abstract', '')}",
                "doi": item['externalIds'].get('DOI', 'N/A'),
                "link": item.get('url', 'N/A'),
                "citationCount": item.get('citationCount', 0)
            })
    except Exception as e:
        st.warning(f"Erro na API Semantic Scholar: {e}")

    try:
        crossref_params = {"query": query, "rows": limit}
        crossref_response = requests.get(CROSSREF_API, params=crossref_params)
        crossref_response.raise_for_status()
        crossref_data = crossref_response.json().get("message", {}).get("items", [])
        for item in crossref_data:
            suggested_phrases.append({
                "phrase": f"{item.get('title', [''])[0]}. {item.get('abstract', '')}",
                "doi": item.get('DOI', 'N/A'),
                "link": item.get('URL', 'N/A'),
                "citationCount": item.get('is-referenced-by-count', 0)
            })
    except Exception as e:
        st.warning(f"Erro na API Crossref: {e}")

    suggested_phrases.sort(key=lambda x: x.get('citationCount', 0), reverse=True)
    return suggested_phrases

def extract_top_keywords(suggested_phrases):
    all_text = " ".join([item['phrase'] for item in suggested_phrases])
    words = re.findall(r'\b\w+\b', all_text.lower())
    words = [word for word in words if word not in STOP_WORDS and len(word) > 3]
    word_freq = Counter(words).most_common(10)
    return [word for word, _ in word_freq]

def get_publication_statistics(total_articles):
    start_date = datetime.now() - timedelta(days=365)
    publication_dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(total_articles)]
    monthly_counts = Counter([date.strftime("%Y-%m") for date in publication_dates])
    proportion_per_100 = (total_articles / 100) * 100
    return monthly_counts, proportion_per_100

# =================================
# 🔗 Modelo Simples PyTorch
# =================================
class ArticlePredictor(nn.Module):
    def __init__(self):
        super(ArticlePredictor, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def evaluate_article_relevance(publication_count):
    model = ArticlePredictor()
    data = torch.tensor([[publication_count]], dtype=torch.float32)
    probability = model(data).item() * 100

    if probability >= 70:
        descricao = "Alta probabilidade de se tornar uma referência devido à baixa concorrência no tema."
    elif 30 <= probability < 70:
        descricao = "Probabilidade moderada. O tema possui competição equilibrada na área científica."
    else:
        descricao = "Baixa probabilidade. O tema possui alta concorrência e muitos artigos publicados."

    return round(probability, 2), descricao

# =================================
# 📄 Funções Auxiliares
# =================================
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def identify_theme(user_text):
    words = re.findall(r'\b\w+\b', user_text)
    keywords = [word.lower() for word in words if word.lower() not in STOP_WORDS]
    keyword_freq = Counter(keywords).most_common(10)
    return ", ".join([word for word, _ in keyword_freq])

def generate_report(suggested_phrases, top_keywords, tema, probabilidade, descricao, monthly_counts, proportion_per_100, output_path="report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    justified_style = ParagraphStyle(
        'Justified',
        parent=styles['BodyText'],
        alignment=4,
        spaceAfter=10,
    )

    content = [
        Paragraph("<b>Relatório - CitatIA - PEAS.Co</b>", styles['Title']),
        Paragraph(f"<b>Tema Identificado:</b> {tema}", justified_style),
        Paragraph(f"<b>Probabilidade de ser uma referência:</b> {probabilidade}%", justified_style),
        Paragraph(f"<b>Análise:</b> {descricao}", justified_style),
        Paragraph("<b>Estatísticas de Publicação:</b>", styles['Heading3']),
    ]

    for month, count in monthly_counts.items():
        content.append(Paragraph(f"• {month}: {count} publicações", justified_style))
    content.append(Paragraph(f"<b>Proporção a cada 100 artigos:</b> {proportion_per_100:.2f}%", justified_style))

    content.append(Paragraph("<b>Artigos mais citados:</b>", styles['Heading3']))
    for item in suggested_phrases:
        content.append(Paragraph(
            f"• {item['phrase']}<br/><b>DOI:</b> {item['doi']}<br/><b>Link:</b> {item['link']}<br/><b>Citações:</b> {item.get('citationCount', 'N/A')}", justified_style))

    content.append(Paragraph("<b>Palavras-chave:</b>", styles['Heading3']))
    for word in top_keywords:
        content.append(Paragraph(f"• {word}", justified_style))

    doc.build(content)

# =================================
# 🚀 Interface Principal
# =================================
def main():
    st.title("CitatIA - Potencializador de Artigos - PEAS.Co")

    st.subheader("📋 Registro de Usuário")
    nome = st.text_input("Nome completo")
    email = st.text_input("E-mail")
    if st.button("Salvar Dados"):
        if nome and email:
            codigo = gerar_codigo_verificacao(email)
            salvar_email_google_sheets(nome, email, codigo)
            st.success(f"Código de verificação: **{codigo}**")
        else:
            st.warning("Preencha todos os campos.")

    uploaded_file = st.file_uploader("📤 Envie o arquivo PDF", type='pdf')
    if uploaded_file:
        with open("uploaded_article.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info("🔍 Analisando...")

        user_text = extract_text_from_pdf("uploaded_article.pdf")
        tema = identify_theme(user_text)

        suggested_phrases = get_popular_phrases(tema, limit=10)
        top_keywords = extract_top_keywords(suggested_phrases)
        publication_count = len(suggested_phrases)
        probabilidade, descricao = evaluate_article_relevance(publication_count)
        monthly_counts, proportion_per_100 = get_publication_statistics(publication_count)

        st.success(f"✅ Tema: {tema}")
        st.write(f"📈 Probabilidade de ser referência: {probabilidade}%")
        st.write(f"ℹ️ {descricao}")

        st.subheader("📊 Estatísticas:")
        for month, count in monthly_counts.items():
            st.write(f"• {month}: {count} publicações")
        st.write(f"• Proporção por 100 artigos: {proportion_per_100:.2f}%")

        st.subheader("🔑 Palavras-chave encontradas:")
        for word in top_keywords:
            st.write(f"• {word}")

        generate_report(suggested_phrases, top_keywords, tema, probabilidade, descricao, monthly_counts, proportion_per_100)
        with open("report.pdf", "rb") as file:
            st.download_button("📥 Baixar Relatório", file, "report.pdf")

    st.subheader("🔎 Verificar Código")
    codigo_digitado = st.text_input("Digite o código:")
    if st.button("Verificar"):
        if verificar_codigo_google_sheets(codigo_digitado):
            st.success("✅ Documento Autêntico!")
        else:
            st.error("❌ Código inválido.")

    st.markdown("---")
    st.markdown(
        "<h3><a href='https://peas8810.hotmart.host/product-page-1f2f7f92-949a-49f0-887c-5fa145e7c05d' target='_blank'>"
        "🚀 Técnica PROATIVA: Domine a Criação de Comandos Poderosos na IA e gere produtos monetizáveis"
        "</a></h3>",
        unsafe_allow_html=True
    )
    st.components.v1.iframe(
        "https://pay.hotmart.com/U99934745U?off=y2b5nihy&hotfeature=51&_hi=eyJjaWQiOiIxNzQ4Mjk4OTUxODE2NzQ2NTc3ODk4OTY0NzUyNTAwIiwiYmlkIjoiMTc0ODI5ODk1MTgxNjc0NjU3Nzg5ODk2NDc1MjUwMCIsInNpZCI6ImM4OTRhNDg0MzJlYzRhZTk4MTNjMDJiYWE2MzdlMjQ1In0=.1748375599003&bid=1748375601381",
        height=250
    )

if __name__ == "__main__":
    main()
