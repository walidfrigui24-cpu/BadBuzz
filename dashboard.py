import sys
import asyncio

# إصلاح مشكلة ويندوز
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import nest_asyncio
from datetime import datetime, timedelta

# استيراد مكتبات الذكاء الاصطناعي المحلية
from transformers import pipeline

from api_client import TwitterAPIClient
from youtube_client import YouTubeClient

nest_asyncio.apply()

st.set_page_config(page_title="War Room (Local AI Core)", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0f1419; color: white; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #272c30; }
    div[data-testid="metric-container"] { background-color: #f7f9f9; padding: 15px; border-radius: 5px; border: 1px solid #e1e8ed; }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'Positif': '#17bf63', 'Négatif': '#e0245e', 'Neutre': '#657786'}

# --- 🧠 تحميل الموديل محلياً ---
@st.cache_resource
def load_local_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return sentiment_pipeline

with st.spinner("جارٍ تحميل الدماغ الإلكتروني (AI Model)... يرجى الانتظار..."):
    try:
        ai_pipeline = load_local_model()
        st.sidebar.success("✅ AI Model Loaded (Local)")
    except Exception as e:
        st.error(f"فشل تحميل الموديل: {e}")
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
    except Exception as e:
        return 0.0, "Neutre"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres (Local AI)")
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
        btn_start = st.form_submit_button("🚀 Lancer")

# --- DASHBOARD ---
st.title("🛡️ War Room (Local Advanced AI)")

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
                status_t.update(label="Twitter OK", state="complete")

    # 2. YOUTUBE
    if "YouTube" in source_mode:
        y_client = YouTubeClient()
        y_query = f"{query_main} {query_exact} {query_any}".strip() or "Actualités"
        with st.spinner("Recherche YouTube..."):
            y_results = y_client.search_videos(y_query, limit=limit)
            final_data.extend(y_results)
            st.success(f"YouTube: {len(y_results)} vidéos")

    # 3. ANALYSE LOCALE AVANCÉE
    if final_data:
        df = pd.DataFrame(final_data)
        if 'metrics' not in df.columns: df['metrics'] = 0
        df['metrics'] = pd.to_numeric(df['metrics'], errors='coerce').fillna(0).astype(int)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        st.info(f"Analyse IA Locale en cours ({len(df)} éléments)...")
        
        scores = []
        sentiments = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(df['text']):
            s, l = analyze_local_advanced(str(text))
            scores.append(s)
            sentiments.append(l)
            if i % 10 == 0: progress_bar.progress((i + 1) / len(df))
            
        progress_bar.empty()
        
        df['score'] = scores
        df['sentiment'] = sentiments
        
        st.divider()

        # =========================================================
        #  1. FILTRAGE & KPIs (تم النقل للأعلى حسب الطلب) ⬆️
        # =========================================================
        st.markdown("### 🔍 Filtrage & KPIs")
        
        selected_sentiments = st.multiselect(
            "Filtrer par Sentiment :", 
            ["Positif", "Négatif", "Neutre"], 
            default=["Positif", "Négatif", "Neutre"]
        )
        
        # تطبيق الفلتر على البيانات
        df_filtered = df[df['sentiment'].isin(selected_sentiments)]

        if not df_filtered.empty:
            # عرض المؤشرات (KPIs)
            c1, c2, c3 = st.columns(3)
            c1.metric("Volume Analysé", len(df_filtered))
            c2.metric("Impact Total (Engagement)", f"{df_filtered['metrics'].sum():,}")
            
            neg_vol = len(df_filtered[df_filtered['sentiment'] == 'Négatif'])
            neg_pct = round((neg_vol / len(df_filtered)) * 100, 1) if len(df_filtered) > 0 else 0
            c3.metric("Taux Négativité", f"{neg_pct}%", delta_color="inverse")
            
            st.divider()

            # =========================================================
            #  2. GRAPHIQUES STRATÉGIQUES (ROW 1)
            # =========================================================
            col_detracteurs, col_trend = st.columns(2)

            # --- A. TOP DÉTRACTEURS ---
            with col_detracteurs:
                st.subheader("🚨 Top Auteurs Négatifs")
                # ملاحظة: نستخدم df_filtered لكي يستجيب المبيان للفلتر (إذا أزلت "Négatif" يختفي المبيان)
                detractors_df = df_filtered[df_filtered['sentiment'] == 'Négatif'].copy()
                
                if not detractors_df.empty:
                    detractors_stats = detractors_df.groupby('author')[['metrics']].sum().reset_index()
                    detractors_stats = detractors_stats.sort_values(by='metrics', ascending=False).head(10)
                    
                    fig_detractors = px.bar(
                        detractors_stats,
                        x='metrics',
                        y='author',
                        orientation='h',
                        text='metrics',
                        color_discrete_sequence=['#e0245e'],
                        labels={"metrics": "Impact", "author": ""}
                    )
                    fig_detractors.update_layout(yaxis=dict(autorange="reversed"), height=400)
                    st.plotly_chart(fig_detractors, use_container_width=True)
                else:
                    st.success("Aucun auteur négatif dans la sélection actuelle.")

            # --- B. SOLDE NET 4H ---
            with col_trend:
                st.subheader("📉 Solde Net (4H)")
                df_trend = df_filtered.dropna(subset=['date']).copy()
                df_polar = df_trend[df_trend['sentiment'] != 'Neutre']
                
                if not df_polar.empty:
                    try:
                        df_agg = df_polar.groupby([pd.Grouper(key='date', freq='4H'), 'sentiment']).size().unstack(fill_value=0)
                        if 'Positif' not in df_agg.columns: df_agg['Positif'] = 0
                        if 'Négatif' not in df_agg.columns: df_agg['Négatif'] = 0
                        
                        df_agg['net_score'] = df_agg['Positif'] - df_agg['Négatif']
                        df_agg['trend_label'] = df_agg['net_score'].apply(lambda x: 'Positif' if x >= 0 else 'Négatif')
                        df_agg = df_agg.reset_index()
                        
                        fig_trend = px.bar(
                            df_agg, 
                            x="date", 
                            y="net_score", 
                            color="trend_label", 
                            color_discrete_map=COLOR_MAP,
                            labels={"net_score": "Solde Net", "date": ""}
                        )
                        fig_trend.update_layout(showlegend=False, height=400, bargap=0.1)
                        fig_trend.add_hline(y=0, line_color="white", opacity=0.5)
                        st.plotly_chart(fig_trend, use_container_width=True)
                    except:
                        st.warning("Données temporelles insuffisantes.")
                else:
                    st.info("Pas assez de données polarisées.")

            # =========================================================
            #  3. ANALYSE GLOBALE (ROW 2)
            # =========================================================
            st.divider()
            g1, g2 = st.columns([1, 2])
            
            with g1:
                st.subheader("Répartition")
                fig_pie = px.pie(df_filtered, names='sentiment', color='sentiment', color_discrete_map=COLOR_MAP)
                st.plotly_chart(fig_pie, use_container_width=True)

            with g2:
                st.subheader("Impact vs Sentiment")
                fig_scatter = px.scatter(
                    df_filtered, 
                    x="metrics", 
                    y="score", 
                    color="sentiment", 
                    color_discrete_map=COLOR_MAP, 
                    hover_data=['text', 'author'], 
                    size="metrics", 
                    size_max=40,
                    labels={"metrics": "Impact", "score": "Sentiment"}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # =========================================================
            #  4. TABLEAU DE DONNÉES
            # =========================================================
            st.subheader("📋 Registre des Données")
            disp = df_filtered[['source', 'date', 'author', 'text', 'sentiment', 'metrics', 'score']].copy()
            st.dataframe(
                disp, 
                use_container_width=True, 
                column_config={
                    "metrics": st.column_config.NumberColumn("Impact", format="%d 👁️"),
                    "score": st.column_config.ProgressColumn("Intensité", min_value=-1, max_value=1)
                }
            )
            
        else:
            st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    else:
        st.warning("Aucun résultat trouvé.")
