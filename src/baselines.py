import re
import pandas as pd
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Standard reply to customers
TRIVIAL_REPLY = "Thankyou for reaching out to customer support. Please send us a DM with your query and details so we can have a look at your problem closely!"

# High Risk keywords that need escalation to customer support agent
ESCALATION_KEYWORDS = [
    r"\bfraud\b", r"\bstolen\b", r"\bhacked\b", r"\bunauthorized\b",
    r"\blawyer\b", r"\blegal\b", r"\bcourt\b", r"\bpolice\b",
    r"\bchargeback\b", r"\bbank\b", r"\bdispute\b", r"\bconsumer court\b",
    r"\bfuck\b", r"\bfucking\b", r"\bshit\b", r"\bscam\b", r"\bcheated\b"
]

ESCALATION_REGEX = re.compile("|".join(ESCALATION_KEYWORDS), re.IGNORECASE)

class TrivialBaseline:
    """A baseline model that never escalates to the agent and uses the generic trivial reply"""
    def __init__(self, majority_intent: str = "ORDER_TRACKING_DELAY"):
        self.majority_intent = majority_intent
    
    def predict(self, text: str) -> Dict[str, any]:
        return {
            "predicted_intent": self.majority_intent,
            "should_escalate": False,
            "escalation_reason": None,
            "drafted_reply": TRIVIAL_REPLY,
            "model": "trivial baseline model"
        }

class SimpleBaseline:
    """
    - INTENT Classification
    - Logistic Regression
    - One Nearest Neighbor classifier
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features = 2000, stop_words = 'english', ngram_range = (1, 2))
        self.classifier = LogisticRegression(max_iter = 1000, random_state = 42)
        self.train_texts: List[str] = []
        self.train_replies: List[str] = []
        self.train_vectors: None
        self.is_trained: bool = False

    def fit(self, train_df: pd.DataFrame, text_col: str = 'customer_clean', intent_col = 'gold_intent', reply_col = 'brand_clean'):
        """Trains the vectorizer classifier and indexes historical customer brand reply pairs"""
        clean_df = train_df.dropna(subset = [text_col, intent_col]).copy()
        self.train_texts = clean_df[text_col].tolist()
        self.train_replies = clean_df[reply_col].tolist if reply_col in clean_df.columns else []

        # Logistic Regression and Vectorizer Classifier
        x = self.vectorizer.fit_transofmr(clean_df[text_col])
        y = clean_df[intent_col]
        self.classifier.fit(x, y)

        # Store training vectors for 1 nearest neighbor classifier
        self.train_vectors = x
        self.is_trained = True
        print(f"Simple baseline training complete for {len(clean_df)} across {len(self.classifier.classes_)} samples.")

    def predict(self, text: str) -> Dict[str, any]:
        """Performs prediction"""
        if not self.is_trained:
            raise RuntimeError("Train model first using fit()")
        else:
            # Predict intent using TF and Logistic Regression
            vector = self.vectorizer.transform([text])
            predicted_intent = self.classifier.predict(vector)[0]
            
            # Predict whether escalation needed for regex
            regex_match = ESCALATION_REGEX.search(text)
            should_escalate = bool(regex_match)
            escalation_reason = f"KEYWORD MATCH. {regex_match.group(0) if regex_match else None}"

            # Perform 1-NN classification
            if self.train_vectors is not None and len(self.train_replies) > 0:
                similarities = cosine_similarity(vector, self.train_vectors).flatten()
                best_idx = similarities.argmax()
                drafted_reply = self.train_replies[best_idx]
            else:
                drafted_reply = TRIVIAL_REPLY
        
        return {
            "predicted_intent": predicted_intent,
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
            "drafted_reply": drafted_reply,
            "model": "simple baseline model"
        }