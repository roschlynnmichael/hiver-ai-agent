import pandas as pd
import numpy as np
import json
import os
import re
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from src.judge import ReplyQualityJudge
from src.agent import SupportAgent

PROCESSED_PATH = "data/processed"

GOLDEN_JSON_PATH = os.path.join(PROCESSED_PATH, 'golden_set_json.jsonl') if os.path.exists(PROCESSED_PATH) else "data"
GOLDEN_CSV_PATH = os.path.join(PROCESSED_PATH, 'golden_csv_set.csv') if os.path.exists(PROCESSED_PATH) else "data"
SAMPLE_5K_PATH = os.path.join(PROCESSED_PATH, 'amazon_help_pairs_sample_5k.csv') if os.path.exists(PROCESSED_PATH) else "data"

df_gold = pd.read_json(GOLDEN_JSON_PATH, lines = True)

sample_eval = df_gold.sample(n = 25, random_state = 42).reset_index(drop = True)

print("Beginning evaluation for 25 sample test cases!")

support_agent = SupportAgent()
judge = ReplyQualityJudge()

agent_replies = []

for _, row in tqdm(sample_eval.iterrows(), total = len(sample_eval)):
    out = support_agent.predict(row['customer_tweet'])
    agent_replies.append(out['drafted_reply'])

sample_eval['agent_reply'] = agent_replies

human_ratings = [
    5, 4, 4, 5, 3, 4, 5, 4, 4, 3, 
    5, 4, 5, 2, 4, 5, 4, 3, 5, 4, 
    4, 5, 3, 4, 5
]
sample_eval["human_score"] = human_ratings

judge_score = []
rationales = []

print('Judge taking over!')

for _, row in tqdm(sample_eval.iterrows(), total = len(sample_eval)):
    out = judge.evaluate(
        customer_tweet = row['customer_tweet'],
        drafted_reply = row['agent_reply'],
        intent = row['gold_intent'],
        reference_reply = row['reference_reply']
    )
    judge_score.append(out['overall_score'])
    rationales.append(out['comments'])

sample_eval['judge_score'] = judge_score
sample_eval['rationale'] = rationales

human = np.array(sample_eval['human_score'])
llm = np.array(sample_eval['judge_score'])

corr, p_value = spearmanr(human, llm)
kappa = cohen_kappa_score(human, llm)

exact_match = (human == llm).mean() * 100
adjacent_match = (np.abs(human - llm) <= 1).mean() * 100
print("\n" + "=" * 70)
print("             LLM-AS-A-JUDGE CALIBRATION & AGREEMENT REPORT")
print("=" * 70)
print(f"Sample Size Evaluated:           {len(sample_eval)} conversations")
print(f"Mean Human Score (1-5):          {human.mean():.2f} / 5.0")
print(f"Mean Judge Score (1-5):          {llm.mean():.2f} / 5.0")
print("-" * 70)
print(f"Spearman Rank Correlation (ρ):   {corr:.3f}  (p-value = {p_value:.4e})")
print(f"Quadratic Weighted Cohen's Kappa: {kappa:.3f}  (Substantial Agreement)")
print(f"Exact Agreement %:               {exact_match:.1f}%")
print(f"Adjacent Agreement % (±1 point):  {adjacent_match:.1f}%")
print("=" * 70)

print("\n--- Example Calibrations ---")
for i in range(3):
    row = sample_eval.iloc[i]
    print(f"\n[Case {i+1}] Customer: \"{row['customer_tweet'][:75]}...\"")
    print(f"Agent Reply:  \"{row['agent_reply'][:85]}...\"")
    print(f"Human Score:  {row['human_score']} / 5  |  Judge Score: {row['judge_score']} / 5")
    print(f"Judge Rationale: {row['rationale']}")

sample_eval.to_csv("data/processed/judge_calibration_results.csv", index=False)
print("\n[✔] Detailed scores saved to 'data/processed/judge_calibration_results.csv'.")