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
from transformers import pipeline
from api_client import TwitterAPIClient
from youtube_client import YouTubeClient

nest_asyncio.apply()

# --- CONFIGURATION ---
st.set_page_config(page_title="War Room", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0f1419; color: white; border-radius: 4px; font-weight: bold; border: 1px solid #333; }
    .stButton>button:hover { background-color: #272c30; border-color: #1DA1F2; }
    div[data-testid="metric-container"] { background-color: #f7f9f9; padding: 15px; border-radius: 8px; border-left: 5px solid #1DA1F2; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'Positif': '#17bf63', 'Négatif': '#e0245e', 'Neutre': '#657786'}

# --- CHARGEMENT MODELE (CACHE) ---
@st.cache_resource
def load_local_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

with st.spinner("Chargement IA..."):
    try:
        ai_pipeline = load_local_model()
        st.sidebar.success("IA Prête")
    except:
        st.sidebar.error("Erreur IA")
        ai_pipeline = None

def analyze_local_advanced(text):
    if not ai_pipeline: return 0.0, "Neutre"
    try:
        safe_text = str(text)[:512]
        result = ai_pipeline(safe_text)[0]
        label = result['label']
        score = result['score']
        
        if label.lower() == 'positive': return score, "Positif"
        elif label.lower() == 'negative': return -score, "Négatif"
        else: return 0.0, "Neutre"
    except:
        return 0.0, "Neutre"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    source_mode = st.radio("Source", ["Twitter (X)", "YouTube", "Fusion (Twitter + YouTube)"])
    
    with st.form("search_form"):
        st.subheader("1. Cible")
        query_main = st.text_input("Mots-clés (AND)", placeholder="Ex: Boycott, Scandale")
        query_exact = st.text_input("Phrase Exacte", placeholder="Ex: La direction a démissionné")
        query_any = st.text_input("Optionnels (OR)", placeholder="Ex: Urgent, Alerte")
        query_exclude = st.text_input("Exclusions (NOT)", placeholder="Ex: Sport, Concert")
        
        st.subheader("2. Période")
        d1, d2 = st.columns(2)
        date_start = d1.date_input("Début", datetime.now() - timedelta(days=7))
        date_end = d2.date_input("Fin", datetime.now())

        if "Twitter" in source_mode:
            with st.expander("3. Filtres Avancés (Twitter)"):
                from_accts = st.text_input("Auteur (@)")
                to_accts = st.text_input("Destinataire (@)")
                mention_accts = st.text_input("Mentionnant (@)")
                min_faves = st.number_input("Min Likes", 0, step=10)
                min_retweets = st.number_input("Min Retweets", 0, step=5)
                links_filter = st.radio("Liens", ["Tous", "Exclure", "Inclure"], index=0)
                replies_filter = st.radio("Réponses", ["Tous", "Exclure", "Inclure"], index=0)
        else:
            from_accts, to_accts, mention_accts = "", "", ""
            min_faves, min_retweets = 0, 0
            links_filter, replies_filter = "Tous", "Tous"

        st.subheader("4. Volume")
        limit = st.number_input("Limite (Max)", 10, 5000, 100, step=50)
        
        st.markdown("---") 
        lang = st.selectbox("Langue Cible", ["Tout", "fr", "en", "ar"], index=1)
        
        btn_start = st.form_submit_button("Lancer l'Analyse")

# --- DASHBOARD LOGIC ---
st.title("🛡️ War Room : Tableau de Bord")

# تهيئة Session State
if 'df_main' not in st.session_state:
    st.session_state['df_main'] = pd.DataFrame()

# عند الضغط على الزر، نقوم بالجلب والحفظ
if btn_start:
    final_data = []
    
    if "Twitter" in source_mode:
        t_client = TwitterAPIClient()
        params_t = {
            "all_words": query_main, "exact_phrase": query_exact,
            "any_words": query_any, "none_words": query_exclude,
            "lang": lang, 
            "from_accounts": from_accts, "to_accounts": to_accts,
            "mention_accounts": mention_accts, "min_faves": min_faves, "min_retweets": min_retweets,
            "links_filter": links_filter, "replies_filter": replies_filter,
            "since": date_start.strftime("%Y-%m-%d"), "until": date_end.strftime("%Y-%m-%d")
        }
        status_t = st.status("Twitter...", expanded=True)
        for update in t_client.fetch_tweets_generator(params_t, limit):
            if "error" in update: st.error(update['error']); break
            status_t.update(label=f"Twitter : {update.get('count', 0)}")
            if update.get('finished'):
                final_data.extend(update['data'])
                status_t.update(label="Twitter OK", state="complete")

    if "YouTube" in source_mode:
        y_client = YouTubeClient()
        y_query = f"{query_main} {query_exact} {query_any}".strip() or "Actualités"
        with st.spinner("YouTube..."):
            y_results = y_client.search_videos(y_query, limit=limit)
            final_data.extend(y_results)
            st.success(f"YouTube : {len(y_results)} vidéos")

    if final_data:
        df = pd.DataFrame(final_data)
        if 'metrics' not in df.columns: df['metrics'] = 0
        df['metrics'] = pd.to_numeric(df['metrics'], errors='coerce').fillna(0).astype(int)
        
        # التنظيف الأولي للتواريخ
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        if df['date'].dt.tz is not None:
             df['date'] = df['date'].dt.tz_localize(None)

        st.info(f"Analyse IA en cours ({len(df)} éléments)...")
        
        scores, sentiments = [], []
        prog = st.progress(0)
        
        for i, text in enumerate(df['text']):
            s, l = analyze_local_advanced(str(text))
            scores.append(s)
            sentiments.append(l)
            if i % 10 == 0: prog.progress((i + 1) / len(df))
        prog.empty()
        
        df['score'] = scores
        df['sentiment'] = sentiments
        
        # حفظ البيانات
        st.session_state['df_main'] = df
        st.rerun()

# --- عرض البيانات من الذاكرة ---
if not st.session_state['df_main'].empty:
    df = st.session_state['df_main'].copy() # نستخدم نسخة لتجنب التعديل على الأصل
    
    # --- SAFETY FIX: تنظيف التواريخ مرة أخرى لضمان التوافق ---
    # هذا الكود سيصلح الخطأ حتى لو كانت البيانات القديمة في الذاكرة تحتوي على Timezone
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
    # ---------------------------------------------------------

    st.divider()

    st.markdown("### Contrôle")
    sel_sentiments = st.multiselect("Filtre Sentiment :", ["Positif", "Négatif", "Neutre"], default=["Positif", "Négatif", "Neutre"])
    df_filtered = df[df['sentiment'].isin(sel_sentiments)]

    if not df_filtered.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("Volume", f"{len(df_filtered)}")
        k2.metric("Impact (Eng.)", f"{df_filtered['metrics'].sum():,}")
        neg_vol = len(df_filtered[df_filtered['sentiment'] == 'Négatif'])
        neg_pct = round((neg_vol / len(df_filtered)) * 100, 1) if len(df_filtered) > 0 else 0
        k3.metric("Négativité", f"{neg_pct}%", delta_color="inverse")
        
        st.divider()

        c_detract, c_trend = st.columns(2)

        with c_detract:
            st.subheader("Top Détracteurs")
            detr_df = df_filtered[df_filtered['sentiment'] == 'Négatif']
            if not detr_df.empty:
                stats = detr_df.groupby('author')[['metrics']].sum().reset_index().sort_values('metrics', ascending=False).head(10)
                fig = px.bar(stats, x='metrics', y='author', orientation='h', text='metrics', color_discrete_sequence=['#e0245e'])
                fig.update_layout(yaxis=dict(autorange="reversed"), height=350, title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("R.A.S")

        with c_trend:
            st.subheader("Solde Net (4H)")
            df_tr = df_filtered.dropna(subset=['date']).copy()
            
            # الآن المقارنة ستنجح بالتأكيد لأننا نظفنا التواريخ بالأعلى
            mask_date = (df_tr['date'] >= pd.Timestamp(date_start)) & (df_tr['date'] <= pd.Timestamp(date_end) + pd.Timedelta(days=1))
            df_tr = df_tr[mask_date]

            df_pol = df_tr[df_tr['sentiment'] != 'Neutre']
            if not df_pol.empty:
                try:
                    agg = df_pol.groupby([pd.Grouper(key='date', freq='4H'), 'sentiment']).size().unstack(fill_value=0)
                    if 'Positif' not in agg: agg['Positif'] = 0
                    if 'Négatif' not in agg: agg['Négatif'] = 0
                    agg['net'] = agg['Positif'] - agg['Négatif']
                    agg['label'] = agg['net'].apply(lambda x: 'Positif' if x >= 0 else 'Négatif')
                    agg = agg.reset_index()
                    
                    fig = px.bar(agg, x="date", y="net", color="label", color_discrete_map=COLOR_MAP)
                    fig.update_traces(width=14400000)
                    fig.update_layout(showlegend=False, height=350, bargap=0)
                    st.plotly_chart(fig, use_container_width=True)
                except: st.warning("Données insuffisantes")
            else: st.info("Pas de données polarisées dans cette période.")

        st.divider()

        g1, g2 = st.columns([1, 2])
        with g1:
            st.subheader("Répartition")
            fig = px.pie(df_filtered, names='sentiment', color='sentiment', color_discrete_map=COLOR_MAP, hole=0.4)
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.subheader("Matrice Impact")
            fig = px.scatter(df_filtered, x="metrics", y="score", color="sentiment", color_discrete_map=COLOR_MAP, 
                            hover_data=['text', 'author'], size="metrics", size_max=50)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Registre")
        disp = df_filtered[['source', 'date', 'author', 'text', 'sentiment', 'metrics', 'score']]
        st.dataframe(disp, use_container_width=True, 
                        column_config={"metrics": st.column_config.NumberColumn("Impact", format="%d"),
                                    "score": st.column_config.ProgressColumn("Score", min_value=-1, max_value=1),
                                    "date": st.column_config.DatetimeColumn("Date", format="DD/MM HH:mm")})
        
    else:
        st.warning("Aucune donnée pour ce filtre.")
else:
    if not btn_start:
        st.info("Bienvenue dans la War Room. Configurez les paramètres à gauche et cliquez sur 'Lancer'.")

