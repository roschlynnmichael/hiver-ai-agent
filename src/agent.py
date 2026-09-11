import re
import json
import ollama as llm
from typing import Dict, Any, List
from src.rag import HistoricalRetriever

MODEL_NAME = 'llama3.1:8b'

PROMPT_TEMPLATE = """
    You are an AI support triage agent for @AmazonHelp. Analyze the incoming customer tweet and provide structured triage decisions.
    From the following INTENTS, you can select exactly one.
    - ORDER_TRACKING_DELAY: Package delay, refund status, delivery time and transit issues.
    - RETURN_REFUND: Return process, replacements, refund status, pickup delays, damaged goods.
    - ACCOUNT_PAYMENT_SECURITY: Unauthorized charges, credit card frauds, locked accounts, OTP issues and stolent funds.
    - PRODUCT_POLICY_INQUIRY: Prime fees, warranty, pricing, product availability, policy questions.
    - SERVICE_COMPLAINT: Rude delivery drivers, customer dissatisfaction, recurring unresolved issues, legal threats.
    - OUT_OF_SCOPE: Praises, greetings, memes, general banter.

    ESCALATION POLICY:
    - should_escalate = true IF:
      * Financial/Fraud/Unauthorized payment risks
      * Customer is extremely angry, uses profanity or threatens with legal threats.
      * Account security issues
    - should_escalate = false IF:
      * Routing Package tracking, standard returns, general policy questions, polite feedback
    
    Customer Tweet: "{customer_tweet}"

    Reponsd only with valid JSON in this format

    {{
        "predicted_intent": "<INTENT_NAME>",
        "confidence": <float 0.0 to 1.0>,
        "should_escalate": <true or false>,
        "escalation_reason": "<1-sentence policy reason if true, else null>"
    }}
"""

REPLY_GENERATION_PROMPT = """
    You are an official customer support representative for @AmazonHelp on Twitter
    Draft a polite, precise, concise and helpful reply to the customer (under 200 characters).

    Customer Tweer: "{customer_tweet}"
    Detected Intent: "{intent}"

    Historical Resolutions:
    {historical_resolutions}

    These are the policies that need to be followed strictly:
    1. Be emphathetic and professional.
    2. If sensitive order details or personally identifiable information is needed, ask them to respond to private DM. Never ask for these confidential details in public.
    3. Never make false or fake promises (for example, do not guarantee refunds in 1 hour).
    4. Direct them to help self-service or ask them to DM for details

    Draft Reply:
"""

class SupportAgent():
    def __init__(self, model_name: str = MODEL_NAME, retriver: HistoricalRetriever = None):
        self.model_name = model_name
        self.retriever = retriver if retriver else HistoricalRetriever()
    
    def triage(self, customer_tweet: str) -> Dict[str, Any]:
        prompt = PROMPT_TEMPLATE.format(customer_tweet = customer_tweet)
        try:
            response = llm.chat(model = self.model_name, messages = [{'role': 'user', 'content': prompt}], format = 'json', options = {'temperature': 0.0})
            data = json.loads(response['message']['content'])
            return {
                'predicted_intent': data.get('predicted_intent'),
                'should_escalate': data.get('should_escalate'),
                'escalation_reason': data.get('escalation_reason'),
                'confidence': float(data.get('confidence'))
            }
        except Exception as e:
            print(f"An Error as occured: {e}")
    
    def generate_reply(self, customer_tweet: str, intent: str, should_escalate: bool, escalation_reason: str) -> str:
        if should_escalate:
            return f"I understand this is urgent and requires immediate attention ({escalation_reason}) I am escalating your case directly to a senior specialist. Please DM us your details so we can assist you right away."
        else:
            context_matches = self.retriever.retrieve(customer_tweet, top_k = 2)
            history_text = "\n".join([f"- Example: {m['historical_reply']}" for m in context_matches])
            prompt = REPLY_GENERATION_PROMPT.format(customer_tweet = customer_tweet, intent = intent, historical_resolutions = history_text)
            try:
                response = llm.chat(model = self.model_name, messages = [{'role': 'user', 'content': prompt}], options = {'temperature': 0.2})
                clean_reply = re.sub(r"^(Here'?s a draft reply:?|Draft reply:?|Here is a draft:?)\s*", "", response['message']['content'], flags=re.IGNORECASE).strip().replace('"', '')
                return clean_reply
            except Exception as e:
                print(f'An exception as occured: {e}')
                return "I am having trouble processing your request right now. Please DM us your query and details so we can assist you right away"
    
    def predict(self, customer_tweet: str) -> Dict[str, Any]:
        triage_out = self.triage(customer_tweet)
        reply = self.generate_reply(
            customer_tweet = customer_tweet,
            intent = triage_out["predicted_intent"],
            should_escalate = triage_out["should_escalate"],
            escalation_reason = triage_out["escalation_reason"]
        )
        return {
            "predicted_intent": triage_out["predicted_intent"],
            "should_escalate": triage_out["should_escalate"],
            "escalation_reason": triage_out["escalation_reason"],
            "drafted_reply": reply,
            "confidence": triage_out["confidence"],
            "model": f"SupportAgent ({self.model_name})"
        }
