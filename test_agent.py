from src.agent import SupportAgent

agent = SupportAgent()

tweet_1 = "Where is my order? It was supposed to be delivered yesterday."
print("\n--- Test Case 1 (Standard) ---")
res_1 = agent.predict(tweet_1)
print("Customer:", tweet_1)
print("Intent:  ", res_1["predicted_intent"])
print("Escalate:", res_1["should_escalate"])
print("Reply:   ", res_1["drafted_reply"])

tweet_2 = "FRAUD! Someone hacked my account and charged 500 dollars to my credit card! I'm calling the police!"
print("\n--- Test Case 2 (High Risk) ---")
res_2 = agent.predict(tweet_2)
print("Customer:", tweet_2)
print("Intent:  ", res_2["predicted_intent"])
print("Escalate:", res_2["should_escalate"])
print("Reason:  ", res_2["escalation_reason"])
print("Reply:   ", res_2["drafted_reply"])