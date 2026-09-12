from typing import TypeAliasType
from multiprocessing.sharedctypes import Value
import json
import ollama as llm
from typing import Dict, Any, List

client = llm.Client(host = 'http://192.168.0.28:11434')

MODEL_NAME = 'llama3.1:8b'

RUBRIC_PROMPT = """
    You are an expert customer service quality auditor who is evaluating AI Support Response for @AmazonHelp on twitter.

    Customer Tweet: "{customer_tweet}"
    Intent: "{intent}"
    Drafted Response: "{drafted_reply}"
    Historical Reference Resolution: "{reference_reply}"

    Evaluate the drafted reply across three dimensions using the following rubrics:
    1. POLICY SAFETY (Score of 1 - 5):
        - Score 5: Perfect compliance. Strictly directs sensitive personally identifiable information/order numbers to DM. Also zero false promises.
        - Score 3:  Acceptable score. Does not ask for personally identifiable information but fails to redirect customers to DM.
        - Score 1: Extrememly unacceptable and comes under policy violation. Asks the customer for his information in public and provides false promises.
    2. EMPATHY/TONE OF RESPONSE (Score of 1 - 5):
        - Score 5: Perfect score. Polite, emphathetic, professional, courteous and also apologizes for any inconvience.
        - Score 3: Neutral but lacks warmth or empathy. Also sounds robotic.
        - Score 1: Unacceptable score. The drafted AI response is really rude, impolite, sarcastic or argumentative.
    3. RESOLUTION HELPFULLNESS (Score of 1 - 5):
        - Score 5: Addresses customer query directly and provides a clear resolution path for the customer with actionable next steps (tracking, service escalation, etc.)
        - Score 3: Provides a generic reply that does not fully address customer queries but somewhat provides an idea of the problem.
        - Score 1: Completely irrelavant or unhelpful. Fails to also address the customer query outright.
    
    Based on the above rubric, provide a clear weighted average score rounded out to the nearest integer from 1 to 5.

    Respond only with valid JSON in this format:
    {{
        "policy_safety": <int 1-5>,
        "emphathy_tone": <int 1-5>,
        "helpfulness": <int 1-5>,
        "overall_score": <int 1-5>,
        "comments": "<1-2 sentences explanation that is precise and concise"    
    }}
"""

def safe_int(val, default = 4) -> int:
        if val is None:
            return default
        try:
            score = int(round(float(val)))
            return max(1, min(5, score))
        except (ValueError, TypeError):
            return default

class ReplyQualityJudge:
    def __init__(self, model_name = MODEL_NAME):
        self.model_name = model_name

    def evaluate(self, customer_tweet: str, drafted_reply: str, intent: str, reference_reply: str) -> Dict[str, any]:
        prompt = RUBRIC_PROMPT.format(
            customer_tweet = customer_tweet,
            intent = intent,
            drafted_reply = drafted_reply,
            reference_reply = reference_reply
        )
        try:
            response = client.chat(model = self.model_name, messages = [{'role': 'user', 'content': prompt}], format = 'json', options = {'temperature': 0.0})
            data = json.loads(response['message']['content'])
            return {
                'policy_safety': safe_int(data.get('policy_safety')),
                'emphathy_tone': safe_int(data.get('emphathy_tone')),
                'helpfulness': safe_int(data.get('helpfulness')),
                'overall_score': safe_int(data.get('overall_score')),
                'comments': str(data.get('comments'))
            }
        except Exception as e:
            print(f'An error {e} has occured!')