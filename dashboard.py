import sys
import asyncio

# Correctif pour Windows
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

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="War Room Analytics", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0f1419; color: white; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #272c30; }
    div[data-testid="metric-container"] { background-color: #f7f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e1e8ed; }
    .critic-card { background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 5px solid #e0245e; margin-bottom: 10px; }
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
    """
    Analyse optimisée par lots (Batch Processing).
    Envoie 10 textes à la fois pour accélérer le processus x10.
    """
    if not HF_API_KEY: return [("0.0", "Neutre")] * len(texts)
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    results = []
    
    # Taille du lot (Batch Size)
    BATCH_SIZE = 10 
    total = len(texts)
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total, BATCH_SIZE):
        # Préparation du lot
        batch_texts = texts[i : i + BATCH_SIZE]
        # Troncature pour l'API
        clean_batch = [str(t)[:512] for t in batch_texts]
        
        payload = {"inputs": clean_batch}
        batch_results = []
        
        # Logique de Retry (3 essais par lot)
        success = False
        for attempt in range(3):
            try:
                response = requests.post(API_URL_ROUTER, headers=headers, json=payload, timeout=20)
                
                # Cas 1: Succès
                if response.status_code == 200:
                    data = response.json()
                    # L'API renvoie une liste de listes (une liste de scores par texte)
                    if isinstance(data, list) and len(data) == len(batch_texts):
                        for item in data:
                            # item est une liste de dicts [{'label': 'POS', 'score': 0.9}, ...]
                            if isinstance(item, list):
                                scores = {x['label']: x['score'] for x in item}
                                p, n, z = scores.get('positive', 0), scores.get('negative', 0), scores.get('neutral', 0)
                                
                                if p > n and p > z: batch_results.append((p, "Positif"))
                                elif n > p and n > z: batch_results.append((-n, "Négatif"))
                                else: batch_results.append((0.1, "Neutre"))
                            else:
                                batch_results.append((0.0, "Neutre")) # Erreur format
                        success = True
                        break
                
                # Cas 2: Modèle en chargement
                elif "loading" in response.text.lower():
                    status_text.warning(f"⏳ IA en chauffe... (Essai {attempt+1}/3)")
                    time.sleep(3)
                    continue
                
                # Cas 3: Erreur API (On passe au retry)
                else:
                    time.sleep(1)
            
            except Exception as e:
                time.sleep(1)
        
        # Si échec total du lot, on remplit par "Neutre" pour ne pas décaler les index
        if not success:
            batch_results.extend([(0.0, "Neutre")] * len(batch_texts))
            # On ne s'arrête pas, on continue les autres lots
        
        results.extend(batch_results)
        
        # Mise à jour progression
        current_progress = min((i + BATCH_SIZE) / total, 1.0)
        progress_bar.progress(current_progress)
        status_text.text(f"Analyse IA: {min(i + BATCH_SIZE, total)}/{total} traités...")

    progress_bar.empty()
    status_text.empty()
    return results

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres")
    source_mode = st.radio("Source", ["Twitter (X)", "YouTube", "Fusion (Twitter + YouTube)"])
    
    with st.form("search_form"):
        st.subheader("1. Mots-clés")
        query_main = st.text_input("Mots-clés (AND)", placeholder="Ex: Banque Crise")
        query_exact = st.text_input("Phrase exacte")
        query_any = st.text_input("N'importe lequel (OR)")
        query_exclude = st.text_input("Mots à exclure (NOT)")
        
        st.subheader("2. Période")
        d1, d2 = st.columns(2)
        date_start = d1.date_input("Début", datetime.now() - timedelta(days=7))
        date_end = d2.date_input("Fin", datetime.now())

        if "Twitter" in source_mode:
            with st.expander("3. Filtres Avancés"):
                from_accts = st.text_input("De (@)")
                to_accts = st.text_input("À (@)")
                mention_accts = st.text_input("Mentionnant (@)")
                min_faves = st.number_input("Min Likes", 0)
                min_retweets = st.number_input("Min Retweets", 0)
                links_filter = st.radio("Liens", ["Tous", "Exclure", "Inclure"], index=0)
                replies_filter = st.radio("Réponses", ["Tous", "Exclure", "Inclure"], index=0)
                lang = st.selectbox("Langue", ["Tout", "fr", "en", "ar"], index=1)
        else:
            from_accts, to_accts, mention_accts = "", "", ""
            min_faves, min_retweets = 0, 0
            links_filter, replies_filter, lang = "Tous", "Tous", "Tout"

        st.subheader("4. Volume")
        limit = st.number_input("Limite", 10, 5000, 100, step=50)
        btn_start = st.form_submit_button("Lancer")

# --- MAIN ---
st.title("🛡️ War Room Analytics")

if btn_start:
    final_data = []
    
    # TWITTER
    if "Twitter" in source_mode:
        t_client = TwitterAPIClient()
        params_t = {
            "all_words": query_main, "exact_phrase": query_exact,
            "any_words": query_any, "none_words": query_exclude,
            "lang": lang, "from_accounts": from_accts, "to_accounts": to_accts,
            "mention_accounts": mention_accts, "min_faves": min_faves, "min_retweets": min_retweets,
            "links_filter": links_filter, "replies_filter": replies_filter,
            "since": date_start.strftime("%Y-%m-%d"), "until": date_end.strftime("%Y-%m-%d")
        }
        status_t = st.status("Extraction Twitter...", expanded=True)
        for update in t_client.fetch_tweets_generator(params_t, limit):
            if "error" in update: st.error(update['error']); break
            status_t.update(label=f"Twitter: {update.get('count', 0)} tweets")
            if update.get('finished'):
                final_data.extend(update['data'])
                status_t.update(label="Twitter OK", state="complete")

    # YOUTUBE
    if "YouTube" in source_mode:
        y_client = YouTubeClient()
        y_query = f"{query_main} {query_exact} {query_any}".strip() or "Actualités"
        with st.spinner("Recherche YouTube..."):
            y_results = y_client.search_videos(y_query, limit=limit)
            final_data.extend(y_results)
            st.success(f"YouTube: {len(y_results)} vidéos")

    # ANALYSE
    if final_data:
        df = pd.DataFrame(final_data)
        if 'metrics' not in df.columns: df['metrics'] = 0
        df['metrics'] = pd.to_numeric(df['metrics'], errors='coerce').fillna(0).astype(int)

        # APPEL FONCTION BATCH OPTIMISÉE
        st.info(f"Analyse IA Rapide ({len(df)} éléments)...")
        results_ia = analyze_sentiment_batch(df['text'].tolist())
        
        df['score'] = [s[0] for s in results_ia]
        df['sentiment'] = [s[1] for s in results_ia]
        
        st.divider()

        # TOP DÉTRACTEURS
        st.subheader("🚨 Top Détracteurs (Impact)")
        detractors = df[df['sentiment'] == 'Négatif'].sort_values(by='metrics', ascending=False).head(4)
        if not detractors.empty:
            cols = st.columns(len(detractors))
            for i, (_, row) in enumerate(detractors.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="critic-card">
                        <b>@{row['author']}</b><br>Impact: {row['metrics']}<br><small>{row['text'][:50]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("R.A.S (Aucun détracteur majeur)")
            
        st.divider()

        # FILTRAGE POST-ANALYSE
        st.markdown("### 🔍 Filtrage")
        selected_sentiments = st.multiselect("Filtre Sentiment :", ["Positif", "Négatif", "Neutre"], default=["Positif", "Négatif", "Neutre"])
        df_filtered = df[df['sentiment'].isin(selected_sentiments)]

        if not df_filtered.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Volume", len(df_filtered))
            c2.metric("Impact Total", f"{df_filtered['metrics'].sum():,}")
            neg_pct = round((len(df_filtered[df_filtered['sentiment'] == 'Négatif']) / len(df_filtered)) * 100, 1)
            c3.metric("Taux Négativité", f"{neg_pct}%", delta_color="inverse")

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(px.pie(df_filtered, names='sentiment', color='sentiment', color_discrete_map=COLOR_MAP), use_container_width=True)
            with g2:
                st.plotly_chart(px.bar(df_filtered, x='source', y='metrics', color='sentiment', barmode='group', color_discrete_map=COLOR_MAP), use_container_width=True)
                
            st.subheader("Données")
            disp = df_filtered[['source', 'date', 'author', 'text', 'sentiment', 'metrics']].copy()
            disp.columns = ['Source', 'Date', 'Auteur', 'Contenu', 'Sentiment', 'Impact']
            st.dataframe(disp, use_container_width=True, column_config={"Impact": st.column_config.NumberColumn(format="%d 👁️")})
            
        else:
            st.warning("Aucune donnée avec ce filtre.")
    else:
        st.warning("Aucun résultat.")
