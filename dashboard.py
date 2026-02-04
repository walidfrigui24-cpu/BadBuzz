import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import requests
import time
import nest_asyncio
from datetime import datetime, timedelta

# Importation des clients
from api_client import TwitterAPIClient
from youtube_client import YouTubeClient

nest_asyncio.apply()

# Configuration de la page (Mode Professionnel)
st.set_page_config(page_title="War Room Analytics", layout="wide")

# CSS Pro (Sans decoration inutile)
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0f1419; color: white; border-radius: 4px; }
    .stButton>button:hover { background-color: #272c30; }
    div[data-testid="metric-container"] { background-color: #f7f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e1e8ed; }
</style>
""", unsafe_allow_html=True)

# Couleurs Officielles
COLOR_MAP = {'Positif': '#17bf63', 'Négatif': '#e0245e', 'Neutre': '#657786'}

# --- CONFIGURATION IA ---
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    HF_API_KEY = None

API_URL_ROUTER = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"

def analyze_sentiment_batch(texts):
    """Analyse IA via Hugging Face Router avec Retry Logic"""
    if not HF_API_KEY: return ["Neutre"] * len(texts)
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    results = []
    
    # Barre de progression pour l'analyse
    prog = st.progress(0)
    
    for i, text in enumerate(texts):
        # Troncature stricte
        payload = {"inputs": str(text)[:512]}
        sentiment = "Neutre"
        score = 0.0
        
        for _ in range(5): # 5 Tentatives max
            try:
                resp = requests.post(API_URL_ROUTER, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        scores = {item['label']: item['score'] for item in data[0]}
                        p, n, z = scores.get('positive', 0), scores.get('negative', 0), scores.get('neutral', 0)
                        
                        if p > n and p > z: sentiment = "Positif"; score = p
                        elif n > p and n > z: sentiment = "Négatif"; score = -n
                        else: sentiment = "Neutre"; score = 0.1
                        break
                elif "loading" in resp.text:
                    time.sleep(3) # Attente modele
                    continue
                else:
                    break
            except:
                break
        
        results.append((score, sentiment))
        prog.progress((i + 1) / len(texts))
        
    prog.empty()
    return results

# --- SIDEBAR (CONTROLES) ---
with st.sidebar:
    st.header("Configuration de l'Analyse")
    
    # Selection de la source
    source_mode = st.radio("Sources de donnees", ["Twitter (X)", "YouTube", "Fusion (Twitter + YouTube)"])
    
    with st.form("search_form"):
        st.subheader("Filtres Semantiques")
        query_main = st.text_input("Mots-cles principaux (AND)", placeholder="Ex: Banque Crise")
        query_exact = st.text_input("Phrase exacte")
        query_exclude = st.text_input("Mots a exclure (NOT)")
        
        st.subheader("Parametres Techniques")
        limit = st.number_input("Volume cible", 50, 5000, 100, step=50)
        
        # Options specifiques Twitter
        if "Twitter" in source_mode:
            with st.expander("Filtres Avances (Twitter)"):
                lang = st.selectbox("Langue", ["fr", "en", "ar", "Tout"], index=0)
                twitter_since = st.date_input("Date debut", datetime.now() - timedelta(days=7))
                twitter_until = st.date_input("Date fin", datetime.now())
        
        btn_start = st.form_submit_button("Lancer l'extraction")

# --- LOGIQUE PRINCIPALE ---
st.title("War Room Analytics - Dashboard")

if btn_start:
    final_data = []
    
    # 1. Extraction Twitter
    if "Twitter" in source_mode:
        t_client = TwitterAPIClient()
        params = {
            "all_words": query_main, "exact_phrase": query_exact, "none_words": query_exclude,
            "lang": lang if "Twitter" in source_mode else "fr",
            "since": twitter_since.strftime("%Y-%m-%d"), "until": twitter_until.strftime("%Y-%m-%d")
        }
        
        status_t = st.status("Extraction Twitter en cours...", expanded=True)
        for update in t_client.fetch_tweets_generator(params, limit):
            if "error" in update:
                st.error(f"Erreur Twitter: {update['error']}")
                break
            status_t.update(label=f"Twitter: {update.get('count', 0)} tweets recuperes")
            if update.get('finished'):
                final_data.extend(update['data'])
                status_t.update(label="Extraction Twitter terminee", state="complete")

    # 2. Extraction YouTube
    if "YouTube" in source_mode:
        y_client = YouTubeClient()
        # Construction de la requete simple pour YouTube
        y_query = f"{query_main} {query_exact}".strip()
        
        with st.spinner("Recherche YouTube en cours..."):
            y_results = y_client.search_videos(y_query, limit=limit)
            final_data.extend(y_results)
            st.success(f"YouTube: {len(y_results)} videos trouvees")

    # 3. Traitement & Analyse
    if final_data:
        df = pd.DataFrame(final_data)
        
        st.info(f"Analyse IA en cours sur {len(df)} elements...")
        sentiments = analyze_sentiment_batch(df['text'].tolist())
        
        df['score'] = [s[0] for s in sentiments]
        df['sentiment'] = [s[1] for s in sentiments]
        
        # --- VISUALISATION ---
        st.divider()
        
        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Volume Total", len(df))
        neg_vol = len(df[df['sentiment'] == 'Négatif'])
        c2.metric("Volume Négatif", neg_vol)
        if len(df) > 0:
            ratio = round((neg_vol / len(df)) * 100, 1)
            c3.metric("Taux de Négativité", f"{ratio}%")

        # Graphiques
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Repartition par Sentiment")
            fig_pie = px.pie(df, names='sentiment', color='sentiment', color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_g2:
            st.subheader("Distribution par Source")
            fig_bar = px.histogram(df, x='source', color='sentiment', barmode='group', color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tableau de donnees
        st.subheader("Donnees Detailles")
        st.dataframe(df[['source', 'date', 'author', 'text', 'sentiment']], use_container_width=True)
        
    else:
        st.warning("Aucune donnee trouvee pour ces parametres.")