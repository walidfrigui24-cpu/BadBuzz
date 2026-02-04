from youtubesearchpython import VideosSearch
import time
from typing import List, Dict

class YouTubeClient:
    """
    Client YouTube Scraping (Sans API Key).
    Optimise pour la vitesse et la compatibilite avec le format Twitter.
    """
    
    def search_videos(self, query: str, limit: int = 50) -> List[Dict]:
        results = []
        try:
            # Recherche standard
            videos_search = VideosSearch(query, limit=limit)
            
            # Boucle de pagination
            while len(results) < limit:
                response = videos_search.result()
                
                if not response['result']:
                    break
                
                for vid in response['result']:
                    if len(results) >= limit: break
                    
                    # Extraction et nettoyage
                    title = vid.get('title', '')
                    desc = vid.get('descriptionSnippet', [{'text': ''}])[0]['text']
                    channel = vid.get('channel', {'name': 'Inconnu'})['name']
                    
                    # Normalisation objet standard
                    video_obj = {
                        "source": "YouTube",
                        "id": vid.get('id'),
                        "date": vid.get('publishedTime', 'Recent'),
                        "text": f"{title}\n{desc}",
                        "author": channel,
                        "metrics": 0, # Placeholder pour compatibilite
                        "url": vid.get('link', '')
                    }
                    results.append(video_obj)
                
                # Page suivante
                try:
                    videos_search.next()
                    time.sleep(0.2) # Pause anti-ban
                except:
                    break
                    
        except Exception as e:
            print(f"Erreur YouTube: {e}")
            
        return results
