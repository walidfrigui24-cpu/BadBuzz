import sys
import asyncio

# Correctif pour Windows (Event Loop)
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

# Importation des modules locaux
from api_client import TwitterAPIClient
from youtube_client import YouTubeClient

nest_asyncio.apply()

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="War Room Analytics", layout="wide")

# CSS Professionnel (Style Strict & Sombre)
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0f1419; color: white; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #272c30; }
    div[data-testid="metric-container"] { background-color: #f7f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e1e8ed; }
    .critic-card { background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 5px solid #e0245e; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'Positif': '#17bf63', 'Négatif': '#e0245e', 'Neutre': '#657786'}

# --- CONFIGURATION IA (HUGGING FACE) ---
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    HF_API_KEY = None

# Nouveau Endpoint Router
API_URL_ROUTER = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"

def analyze_sentiment_batch(texts):
    """Analyse les sentiments via l'API Hugging Face avec logique de réessai"""
    if not HF_API_KEY: return ["Neutre"] * len(texts)
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    results = []
    
    # Barre de progression
    prog = st.progress(0)
    
    for i, text in enumerate(texts):
        # Troncature pour respecter la limite du modèle
        payload = {"inputs": str(text)[:512]}
        sentiment = "Neutre"
        score = 0.0
        
        # Tentative de connexion (Retry Logic)
        for _ in range(3): 
            try:
                resp = requests.post(API_URL_ROUTER, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        # Extraction du score le plus élevé
                        scores = {item['label']: item['score'] for item in data[0]}
                        p, n, z = scores.get('positive', 0), scores.get('negative', 0), scores.get('neutral', 0)
                        
                        if p > n and p > z: sentiment = "Positif"; score = p
                        elif n > p and n > z: sentiment = "Négatif"; score = -n
                        else: sentiment = "Neutre"; score = 0.1
                        break
                elif "loading" in resp.text.lower():
                    time.sleep(3) # Attente si le modèle charge
                    continue
                else:
                    break
            except:
                break
        
        results.append((score, sentiment))
        prog.progress((i + 1) / len(texts))
        
    prog.empty()
    return results

# --- BARRE LATÉRALE (FILTRES AVANCÉS) ---
with st.sidebar:
    st.header("Paramètres d'Analyse")
    
    # 1. Sélection de la Source
    source_mode = st.radio("Source de données", ["Twitter (X)", "YouTube", "Fusion (Twitter + YouTube)"])
    
    with st.form("search_form"):
        # 2. Sémantique
        st.subheader("1. Mots-clés")
        query_main = st.text_input("Mots-clés (AND)", placeholder="Ex: Banque Crise")
        query_exact = st.text_input("Phrase exacte")
        query_any = st.text_input("N'importe lequel (OR)")
        query_exclude = st.text_input("Mots à exclure (NOT)")
        
        # 3. Période (Date Picker Direct)
        st.subheader("2. Période")
        d1, d2 = st.columns(2)
        date_start = d1.date_input("Début", datetime.now() - timedelta(days=7))
        date_end = d2.date_input("Fin", datetime.now())

        # 4. Filtres Avancés Twitter
        if "Twitter" in source_mode:
            with st.expander("3. Filtres Avancés (Comptes & Métriques)"):
                st.caption("Ciblage")
                from_accts = st.text_input("De (@)")
                to_accts = st.text_input("À (@)")
                mention_accts = st.text_input("Mentionnant (@)")
                
                st.caption("Engagement Min.")
                min_faves = st.number_input("Min Likes", 0)
                min_retweets = st.number_input("Min Retweets", 0)
                
                st.caption("Filtres Techniques")
                links_filter = st.radio("Liens", ["Tous", "Exclure", "Inclure"], index=0)
                replies_filter = st.radio("Réponses", ["Tous", "Exclure", "Inclure"], index=0)
                lang = st.selectbox("Langue", ["Tout", "fr", "en", "ar"], index=1)
        else:
            # Valeurs par défaut pour éviter les erreurs si Twitter est désactivé
            from_accts, to_accts, mention_accts = "", "", ""
            min_faves, min_retweets = 0, 0
            links_filter, replies_filter, lang = "Tous", "Tous", "Tout"

        # 5. Volume
        st.subheader("4. Volume")
        limit = st.number_input("Limite par source", 10, 5000, 100, step=50)
        
        btn_start = st.form_submit_button("🚀 Lancer l'Analyse")

# --- TABLEAU DE BORD PRINCIPAL ---
st.title("🛡️ War Room Analytics")

if btn_start:
    final_data = []
    
    # --- 1. EXTRACTION TWITTER ---
    if "Twitter" in source_mode:
        t_client = TwitterAPIClient()
        params_t = {
            "all_words": query_main, "exact_phrase": query_exact,
            "any_words": query_any, "none_words": query_exclude,
            "lang": lang,
            "from_accounts": from_accts, "to_accounts": to_accts, "mention_accounts": mention_accts,
            "min_faves": min_faves, "min_retweets": min_retweets,
            "links_filter": links_filter, "replies_filter": replies_filter,
            "since": date_start.strftime("%Y-%m-%d"), "until": date_end.strftime("%Y-%m-%d")
        }
        
        status_t = st.status("Extraction Twitter en cours...", expanded=True)
        for update in t_client.fetch_tweets_generator(params_t, limit):
            if "error" in update:
                st.error(f"Erreur Twitter: {update['error']}")
                break
            status_t.update(label=f"Twitter: {update.get('count', 0)} tweets récupérés")
            if update.get('finished'):
                final_data.extend(update['data'])
                status_t.update(label="Extraction Twitter terminée", state="complete")

    # --- 2. EXTRACTION YOUTUBE ---
    if "YouTube" in source_mode:
        y_client = YouTubeClient()
        y_query = f"{query_main} {query_exact} {query_any}".strip()
        if not y_query: y_query = "Actualités"
        
        with st.spinner("Recherche YouTube en cours..."):
            y_results = y_client.search_videos(y_query, limit=limit)
            final_data.extend(y_results)
            st.success(f"YouTube: {len(y_results)} vidéos trouvées")

    # --- 3. TRAITEMENT & ANALYSE ---
    if final_data:
        df = pd.DataFrame(final_data)
        
        # Nettoyage et conversion metrics
        if 'metrics' not in df.columns: df['metrics'] = 0
        df['metrics'] = pd.to_numeric(df['metrics'], errors='coerce').fillna(0).astype(int)

        st.info(f"Analyse IA en cours sur {len(df)} éléments...")
        sentiments = analyze_sentiment_batch(df['text'].tolist())
        
        df['score'] = [s[0] for s in sentiments]
        df['sentiment'] = [s[1] for s in sentiments]
        
        st.divider()

        # --- A. SECTION: TOP DÉTRACTEURS (IMPACT) ---
        # Identification des auteurs négatifs avec le plus grand impact
        st.subheader("🚨 Top Détracteurs (Alerte Impact)")
        
        detractors = df[df['sentiment'] == 'Négatif'].sort_values(by='metrics', ascending=False).head(4)
        
        if not detractors.empty:
            cols = st.columns(len(detractors))
            for i, (_, row) in enumerate(detractors.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="critic-card">
                        <b>@{row['author']}</b><br>
                        Impact: {row['metrics']}<br>
                        <small>{row['text'][:60]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("Aucun détracteur majeur détecté pour le moment.")
            
        st.divider()

        # --- B. FILTRAGE DYNAMIQUE (POST-ANALYSE) ---
        st.markdown("### 🔍 Filtrage des Résultats")
        selected_sentiments = st.multiselect(
            "Afficher les sentiments :",
            options=["Positif", "Négatif", "Neutre"],
            default=["Positif", "Négatif", "Neutre"]
        )
        
        # Application du filtre
        df_filtered = df[df['sentiment'].isin(selected_sentiments)]

        if not df_filtered.empty:
            
            # --- C. KPIs ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Volume Filtré", len(df_filtered))
            
            engagement_total = df_filtered['metrics'].sum()
            c2.metric("Impact Total (Engagement)", f"{engagement_total:,}")
            
            neg_vol = len(df_filtered[df_filtered['sentiment'] == 'Négatif'])
            ratio_neg = round((neg_vol / len(df_filtered)) * 100, 1) if len(df_filtered) > 0 else 0
            c3.metric("Taux de Négativité", f"{ratio_neg}%", delta_color="inverse")

            # --- D. GRAPHIQUES ---
            g1, g2 = st.columns(2)
            with g1:
                st.subheader("Répartition Sentimentale")
                fig = px.pie(df_filtered, names='sentiment', color='sentiment', color_discrete_map=COLOR_MAP)
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                st.subheader("Impact par Source")
                fig = px.bar(df_filtered, x='source', y='metrics', color='sentiment', barmode='group', color_discrete_map=COLOR_MAP)
                st.plotly_chart(fig, use_container_width=True)
                
            # --- E. TABLEAU DE DONNÉES (AVEC IMPACT) ---
            st.subheader("📋 Registre des Données")
            
            # Sélection et renommage des colonnes pour l'affichage
            display_df = df_filtered[['source', 'date', 'author', 'text', 'sentiment', 'metrics']].copy()
            display_df.columns = ['Source', 'Date', 'Auteur', 'Contenu', 'Sentiment', 'Impact (Metrics)']
            
            st.dataframe(
                display_df, 
                use_container_width=True,
                column_config={
                    "Impact (Metrics)": st.column_config.NumberColumn(
                        "Impact",
                        help="Likes + Retweets (Twitter) ou Vues (YouTube)",
                        format="%d 👁️"
                    )
                }
            )
            
        else:
            st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
            
    else:
        st.warning("Aucun résultat trouvé. Veuillez élargir vos critères de recherche.")
