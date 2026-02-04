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

from api_client import TwitterAPIClient
from youtube_client import YouTubeClient

nest_asyncio.apply()

st.set_page_config(page_title="War Room Analytics", layout="wide")

# CSS Pro (Style Strict)
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0f1419; color: white; border-radius: 4px; }
    .stButton>button:hover { background-color: #272c30; }
    div[data-testid="metric-container"] { background-color: #f7f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e1e8ed; }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'Positif': '#17bf63', 'Négatif': '#e0245e', 'Neutre': '#657786'}

# --- IA CONFIGURATION ---
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    HF_API_KEY = None

API_URL_ROUTER = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"

def analyze_sentiment_batch(texts):
    if not HF_API_KEY: return ["Neutre"] * len(texts)
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    results = []
    prog = st.progress(0)
    
    for i, text in enumerate(texts):
        payload = {"inputs": str(text)[:512]}
        sentiment = "Neutre"
        score = 0.0
        
        for _ in range(3): # Retry logic
            try:
                resp = requests.post(API_URL_ROUTER, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        scores = {item['label']: item['score'] for item in data[0]}
                        p, n, z = scores.get('positive', 0), scores.get('negative', 0), scores.get('neutral', 0)
                        
                        if p > n and p > z: sentiment = "Positif"; score = p
                        elif n > p and n > z: sentiment = "Négatif"; score = -n
                        else: sentiment = "Neutre"; score = 0.1
                        break
                elif "loading" in resp.text.lower():
                    time.sleep(3)
                    continue
                else:
                    break
            except:
                break
        
        results.append((score, sentiment))
        prog.progress((i + 1) / len(texts))
        
    prog.empty()
    return results

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Parametres d'Analyse")
    
    # 1. Source
    source_mode = st.radio("Source de donnees", ["Twitter (X)", "YouTube", "Fusion (Twitter + YouTube)"])
    
    with st.form("search_form"):
        # 2. Semantique
        st.subheader("Filtres Semantiques")
        query_main = st.text_input("Mots-cles (AND)", placeholder="Ex: Banque Crise")
        query_exact = st.text_input("Phrase exacte")
        query_any = st.text_input("N'importe lequel (OR)")
        query_exclude = st.text_input("Mots a exclure (NOT)")
        
        # 3. Periode (VISIBLE DIRECTEMENT - PAS DE DROPDOWN)
        st.subheader("Periode d'Analyse")
        d1, d2 = st.columns(2)
        date_start = d1.date_input("Debut", datetime.now() - timedelta(days=7))
        date_end = d2.date_input("Fin", datetime.now())

        # 4. Twitter Advanced (Dans Expander pour gagner de la place, sauf les dates)
        if "Twitter" in source_mode:
            with st.expander("Filtres Avances Twitter (Comptes & Metriques)"):
                st.caption("Ciblage par comptes")
                from_accts = st.text_input("De (@)")
                to_accts = st.text_input("A (@)")
                mention_accts = st.text_input("Mentionnant (@)")
                
                st.caption("Seuils d'engagement")
                min_faves = st.number_input("Min Likes", 0)
                min_retweets = st.number_input("Min Retweets", 0)
                
                st.caption("Filtres techniques")
                links_filter = st.radio("Liens", ["Tous", "Exclure", "Inclure"], index=0)
                replies_filter = st.radio("Reponses", ["Tous", "Exclure", "Inclure"], index=0)
                lang = st.selectbox("Langue", ["Tout", "fr", "en", "ar"], index=1)
        else:
            # Valeurs par defaut si Twitter inactif
            from_accts, to_accts, mention_accts = "", "", ""
            min_faves, min_retweets = 0, 0
            links_filter, replies_filter, lang = "Tous", "Tous", "Tout"

        # 5. Volume
        st.subheader("Volume")
        limit = st.number_input("Limite par source", 10, 5000, 100, step=50)
        
        btn_start = st.form_submit_button("Lancer l'Analyse")

# --- MAIN DASHBOARD ---
st.title("War Room Analytics")

if btn_start:
    final_data = []
    
    # 1. Execution Twitter
    if "Twitter" in source_mode:
        t_client = TwitterAPIClient()
        # Mapping complet des parametres avances
        params_t = {
            "all_words": query_main, "exact_phrase": query_exact,
            "any_words": query_any, "none_words": query_exclude,
            "lang": lang,
            "from_accounts": from_accts, "to_accounts": to_accts, "mention_accounts": mention_accts,
            "min_faves": min_faves, "min_retweets": min_retweets,
            "links_filter": links_filter, "replies_filter": replies_filter,
            "since": date_start.strftime("%Y-%m-%d"), "until": date_end.strftime("%Y-%m-%d")
        }
        
        status_t = st.status("Traitement Twitter...", expanded=True)
        for update in t_client.fetch_tweets_generator(params_t, limit):
            if "error" in update:
                st.error(f"Twitter Erreur: {update['error']}")
                break
            status_t.update(label=f"Twitter: {update.get('count', 0)} tweets recuperes")
            if update.get('finished'):
                final_data.extend(update['data'])
                status_t.update(label="Twitter Termine", state="complete")

    # 2. Execution YouTube
    if "YouTube" in source_mode:
        y_client = YouTubeClient()
        # Construction requete simple pour YT
        y_query = f"{query_main} {query_exact} {query_any}".strip()
        if not y_query: y_query = "Actualites"
        
        with st.spinner("Recherche YouTube..."):
            y_results = y_client.search_videos(y_query, limit=limit)
            final_data.extend(y_results)
            st.success(f"YouTube: {len(y_results)} videos")

    # 3. Analyse & Rendu
    if final_data:
        df = pd.DataFrame(final_data)
        
        st.info(f"Analyse IA en cours ({len(df)} elements)...")
        sentiments = analyze_sentiment_batch(df['text'].tolist())
        
        df['score'] = [s[0] for s in sentiments]
        df['sentiment'] = [s[1] for s in sentiments]
        
        # --- VISUALISATION ---
        st.divider()
        
        # KPI Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Volume Total", len(df))
        c2.metric("Volume Twitter", len(df[df['source']=='Twitter']))
        c3.metric("Volume YouTube", len(df[df['source']=='YouTube']))
        
        # Charts Row 1
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Sentiment Global")
            fig = px.pie(df, names='sentiment', color='sentiment', color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.subheader("Comparaison Sources")
            fig = px.histogram(df, x='source', color='sentiment', barmode='group', color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig, use_container_width=True)
            
        # Data Table
        st.subheader("Registre des Donnees")
        st.dataframe(df[['source', 'date', 'author', 'text', 'sentiment']], use_container_width=True)
        
    else:
        st.warning("Aucune donnee trouvee. Verifiez vos mots-cles.")
