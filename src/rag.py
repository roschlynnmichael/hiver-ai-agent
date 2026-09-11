import os
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_PATH = "data/processed"
RAW_PATH = "data/raw"

GOLDEN_JSON_PATH = os.path.join(PROCESSED_PATH, 'golden_set_json.jsonl') if os.path.exists(PROCESSED_PATH) else "data"
GOLDEN_CSV_PATH = os.path.join(PROCESSED_PATH, 'golden_csv_set.csv') if os.path.exists(PROCESSED_PATH) else "data"
SAMPLE_5K_PATH = os.path.join(PROCESSED_PATH, 'amazon_help_pairs_sample_5k.csv') if os.path.exists(PROCESSED_PATH) else "data"

MODEL_NAME = "all-MiniLM-L6-v2"

class HistoricalRetriever:
    def __init__(self, data_path: str = SAMPLE_5K_PATH, model_name: str = MODEL_NAME):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.df: pd.DataFrame = pd.DataFrame()
        self.embeddings: np.ndarray = np.array([])
        self.is_indexed = False
    
    def build_index(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found at {self.data_path}")
        else:
            print('Building RAG Vector Index...')
            self.df = pd.read_csv(self.data_path).dropna(subset = ['customer_clean', 'brand_clean']).reset_index(drop = True)
            texts = self.df['customer_clean'].tolist()
            self.embeddings = self.model.encode(texts, batch_size = 64, show_progress_bar = True, normalize_embeddings = True)
            self.is_indexed = True
            print('Building done!')
    
    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.is_indexed:
            self.build_index()
        
        query_vector = self.model.encode([query], normalize_embeddings = True)
        scores = cosine_similarity(query_vector, self.embeddings).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                'similarity_score': float(scores[idx]),
                'historical_query': self.df.iloc[idx]['customer_clean'],
                'historical_reply': self.df.iloc[idx]['brand_clean']
            })
        return results