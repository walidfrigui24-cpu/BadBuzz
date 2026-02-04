from youtubesearchpython import VideosSearch
import time
from typing import List, Dict

class YouTubeClient:
    """
    Client de scraping YouTube rapide (Sans cle API).
    Utilise la bibliotheque youtubesearchpython.
    """
    
    def search_videos(self, query: str, limit: int = 50) -> List[Dict]:
        results = []
        try:
            # Initialisation de la recherche
            videos_search = VideosSearch(query, limit=limit)
            
            # Boucle pour recuperer les resultats
            while len(results) < limit:
                response = videos_search.result()
                
                if not response['result']:
                    break
                
                for vid in response['result']:
                    if len(results) >= limit: break
                    
                    # Normalisation des donnees pour correspondre a Twitter
                    views_text = vid.get('viewCount', {'text': '0'})['text']
                    # Nettoyage simple du compteur de vues (ex: "1.2M views" -> nombre)
                    
                    video_obj = {
                        "source": "YouTube",
                        "id": vid.get('id'),
                        "date": vid.get('publishedTime', 'Recent'), # Date approximative
                        "text": f"{vid.get('title', '')} \n {vid.get('descriptionSnippet', [{'text': ''}])[0]['text']}",
                        "author": vid.get('channel', {'name': 'Inconnu'})['name'],
                        "metrics": 0 # Les vues sont du texte, difficile a convertir proprement sans logique complexe
                    }
                    results.append(video_obj)
                
                # Page suivante
                try:
                    videos_search.next()
                    time.sleep(0.2)
                except:
                    break
                    
        except Exception as e:
            print(f"Erreur YouTube: {e}")
            
        return results