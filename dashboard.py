import sys
import asyncio

# Correctif Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import nest_asyncio
from datetime import datetime, timedelta

# المكتبة المحلية للتحليل
from textblob import TextBlob 

# Importation des clients
from api_client import TwitterAPIClient
from youtube_client import YouTubeClient

nest_asyncio.apply()

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="War Room Analytics (Local)", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0f1419; color: white; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #272c30; }
    div[data-testid="metric-container"] { background-color: #f7f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e1e8ed; }
    .critic-card { background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 5px solid #e0245e; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'Positif': '#17bf63', 'Négatif': '#e0245e', 'Neutre': '#657786'}

# --- FONCTION D'ANALYSE LOCALE (TEXTBLOB) ---
def analyze_local_sentiment(text):
    """
    Analyse ultra-rapide utilisant le CPU local (TextBlob).
    Pas d'API, pas d'attente.
    """
    if not isinstance(text, str): return 0.0, "Neutre"
    
    # Création de l'objet TextBlob
    blob = TextBlob(text)
    
    # Calcul de la polarité (-1 à +1)
    # Note: TextBlob est natif anglais. Pour le français/arabe, c'est approximatif 
    # mais suffisant pour une vue d'ensemble rapide.
    polarity = blob.sentiment.polarity
    
    # Classification
    if polarity > 0.05:
        return polarity, "Positif"
    elif polarity < -0.05:
        return polarity, "Négatif"
    else:
        return polarity, "Neutre"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres (Local Mode ⚡)")
    
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
        
        btn_start = st.form_submit_button("🚀 Lancer l'Analyse")

# --- DASHBOARD ---
st.title("🛡️ War Room Analytics (Local Core)")

if btn_start:
    final_data = []
    
    # 1. TWITTER
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
                status_t.update(label="Twitter Terminé", state="complete")

    # 2. YOUTUBE
    if "YouTube" in source_mode:
        y_client = YouTubeClient()
        y_query = f"{query_main} {query_exact} {query_any}".strip() or "Actualités"
        with st.spinner("Recherche YouTube..."):
            y_results = y_client.search_videos(y_query, limit=limit)
            final_data.extend(y_results)
            st.success(f"YouTube: {len(y_results)} vidéos")

    # 3. ANALYSE LOCALE (INSTANTANÉE)
    if final_data:
        df = pd.DataFrame(final_data)
        if 'metrics' not in df.columns: df['metrics'] = 0
        df['metrics'] = pd.to_numeric(df['metrics'], errors='coerce').fillna(0).astype(int)

        st.info(f"Analyse Locale Rapide ({len(df)} éléments)...")
        
        # --- BOUCLE RAPIDE LOCALE ---
        scores = []
        sentiments = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(df['text']):
            s, l = analyze_local_sentiment(str(text))
            scores.append(s)
            sentiments.append(l)
            if i % 50 == 0: progress_bar.progress((i + 1) / len(df))
            
        progress_bar.empty()
        # -----------------------------
        
        df['score'] = scores
        df['sentiment'] = sentiments
        
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
                        <b>@{row['author']}</b><br>Impact: {row['metrics']}<br><small>{str(row['text'])[:50]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("R.A.S")

        # FILTRAGE
        st.markdown("### 🔍 Filtrage")
        selected_sentiments = st.multiselect("Filtre :", ["Positif", "Négatif", "Neutre"], default=["Positif", "Négatif", "Neutre"])
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
            
            st.subheader("📋 Données")
            disp = df_filtered[['source', 'date', 'author', 'text', 'sentiment', 'metrics']].copy()
            st.dataframe(disp, use_container_width=True, column_config={"metrics": st.column_config.NumberColumn("Impact", format="%d 👁️")})
            
        else:
            st.warning("Aucune donnée.")
    else:
        st.warning("Aucun résultat.")
