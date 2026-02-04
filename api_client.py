import requests
import time
import math
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, Generator

# Recuperation de la cle API Twitter depuis les secrets
try:
    API_KEY = st.secrets["TWITTER_API_KEY"]
except:
    API_KEY = None

API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

class TwitterAPIClient:
    """
    Client API Twitter V7 (Optimise pour la production).
    Gere la pagination temporelle et les erreurs API avec robustesse.
    """
    
    def build_base_query(self, p: Dict[str, Any]) -> str:
        # Construction de la requete booleenne complexe
        parts = []
        
        # 1. Semantique
        if p.get('all_words'): parts.append(p['all_words'])
        if p.get('exact_phrase'): parts.append(f'"{p["exact_phrase"]}"')
        
        if p.get('any_words'): 
            words = p['any_words'].split()
            if len(words) > 1: parts.append(f"({' OR '.join(words)})")
            else: parts.append(words[0])
        
        if p.get('none_words'): 
            for w in p['none_words'].split(): parts.append(f"-{w}")
        
        if p.get('hashtags'): parts.append(p['hashtags'])
        if p.get('lang') and p['lang'] != "Tout": parts.append(f"lang:{p['lang']}")

        # 2. Comptes
        if p.get('from_accounts'): parts.append(f"from:{p['from_accounts'].replace('@', '')}")
        if p.get('to_accounts'): parts.append(f"to:{p['to_accounts'].replace('@', '')}")
        if p.get('mention_accounts'): parts.append(f"@{p['mention_accounts'].replace('@', '')}")

        # 3. Metriques & Filtres
        if p.get('min_faves') and int(p['min_faves']) > 0: parts.append(f"min_faves:{p['min_faves']}")
        if p.get('min_retweets') and int(p['min_retweets']) > 0: parts.append(f"min_retweets:{p['min_retweets']}")
        
        if p.get('links_filter') == "Exclure les liens": parts.append("-filter:links")
        elif p.get('links_filter') == "Uniquement avec liens": parts.append("filter:links")
        if p.get('replies_filter') == "Exclure les reponses": parts.append("exclude:replies")
        elif p.get('replies_filter') == "Uniquement les reponses": parts.append("filter:replies")

        return " ".join(parts)

    def fetch_tweets_generator(self, params: Dict[str, Any], total_limit: int = 50) -> Generator[Dict, None, None]:
        
        if not API_KEY:
            yield {"error": "Cle API Twitter manquante."}
            return

        base_query = self.build_base_query(params)
        headers = {"X-API-Key": API_KEY}
        
        all_tweets = []
        start_time_global = time.time()
        
        # Calcul de la periode
        try:
            d_start = datetime.strptime(params['since'], "%Y-%m-%d")
            d_end = datetime.strptime(params['until'], "%Y-%m-%d")
        except:
            d_start = datetime.now() - timedelta(days=7)
            d_end = datetime.now()

        delta = (d_end - d_start).days
        if delta <= 0: delta = 1
        
        # Repartition equitable (Quota journalier)
        daily_quota = math.ceil(total_limit / delta)
        if daily_quota < 10: daily_quota = 10
        
        current_day = d_start
        
        # Boucle temporelle (Jour par Jour)
        while current_day < d_end:
            if len(all_tweets) >= total_limit: break

            day_str = current_day.strftime("%Y-%m-%d")
            next_day_str = (current_day + timedelta(days=1)).strftime("%Y-%m-%d")
            
            daily_query = f"{base_query} since:{day_str} until:{next_day_str}"
            
            day_tweets = []
            next_cursor = None
            
            # Pagination interne au jour
            while len(day_tweets) < daily_quota:
                
                payload = {"query": daily_query, "limit": 20}
                if next_cursor: payload["cursor"] = next_cursor

                try:
                    response = requests.get(API_URL, params=payload, headers=headers)
                    
                    if response.status_code == 429: # Rate Limit
                        time.sleep(5)
                        continue 
                    
                    if response.status_code == 402: # Quota epuise
                        yield {"error": "Credit API epuise (Erreur 402)."}
                        return

                    if response.status_code != 200:
                        yield {"error": f"Erreur API ({response.status_code})"}
                        break

                    data = response.json()
                    batch = data.get('tweets', [])

                    if not batch: break

                    for t in batch:
                        if any(existing['id'] == t.get('id') for existing in all_tweets): continue
                        
                        author = t.get('author') or {}
                        tweet_obj = {
                            "source": "Twitter",
                            "id": t.get('id'),
                            "date": t.get('createdAt'),
                            "text": t.get('text', ""),
                            "author": author.get('userName', 'Inconnu'),
                            "metrics": t.get('likeCount', 0) + t.get('retweetCount', 0)
                        }
                        all_tweets.append(tweet_obj)
                        day_tweets.append(tweet_obj)

                    yield {
                        "count": len(all_tweets),
                        "target": total_limit,
                        "data": all_tweets,
                        "finished": False
                    }

                    next_cursor = data.get('next_cursor')
                    if not next_cursor or not data.get('has_next_page'): break
                    
                    if len(day_tweets) >= daily_quota: break 

                    time.sleep(0.5) # Pause courte pour performance

                except Exception as e:
                    yield {"error": str(e)}
                    break
            
            current_day += timedelta(days=1)
            time.sleep(1) # Pause inter-jours

        yield {
            "count": len(all_tweets),
            "target": total_limit,
            "data": all_tweets[:total_limit],
            "finished": True
        }