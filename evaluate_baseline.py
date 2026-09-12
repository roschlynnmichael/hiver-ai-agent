import json
import os
import re
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from src.agent import SupportAgent
from src.baselines import TrivialBaseline, SimpleBaseline

PROCESSED_PATH = "data/processed"
RAW_PATH = "data/raw"

GOLDEN_JSON_PATH = os.path.join(PROCESSED_PATH, 'golden_set_json.jsonl') if os.path.exists(PROCESSED_PATH) else "data"
GOLDEN_CSV_PATH = os.path.join(PROCESSED_PATH, 'golden_csv_set.csv') if os.path.exists(PROCESSED_PATH) else "data"
SAMPLE_5K_PATH = os.path.join(PROCESSED_PATH, 'amazon_help_pairs_sample_5k.csv') if os.path.exists(PROCESSED_PATH) else "data"


print(f'Loading the golden json evaluation set from {GOLDEN_JSON_PATH}')

df_gold = pd.read_json(GOLDEN_JSON_PATH, lines = True if GOLDEN_JSON_PATH.endswith('.jsonl') else False)

trivial_model = TrivialBaseline()

print(f'Training simple baseline model on 5k sample data located at {SAMPLE_5K_PATH}')

df_train, df_test = train_test_split(df_gold, test_size=0.25, random_state=42, stratify=df_gold["gold_intent"])
print(f"[*] Training on {len(df_train)} samples, Evaluating on {len(df_test)} unseen test samples...")

simple_model = SimpleBaseline()
simple_model.fit(df_train, text_col = 'customer_tweet', intent_col = 'gold_intent', reply_col = 'reference_reply')

support_agent = SupportAgent()

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer = True)

def evaluate_model(model, df_eval, model_name = 'Model'):
    preds_intent = []
    preds_escalate = []
    rouge_scores = []

    for _, row in df_eval.iterrows():
        out = model.predict(row['customer_tweet'])
        preds_intent.append(out['predicted_intent'])
        preds_escalate.append(out['should_escalate'])

        score = scorer.score(row['reference_reply'], out['drafted_reply'])
        rouge_scores.append(score['rougeL'].fmeasure)

    y_true_intent = df_eval['gold_intent']
    intent_acc = accuracy_score(y_true_intent, preds_intent)
    intent_p, intent_r, intent_f1, _ = precision_recall_fscore_support(y_true_intent, preds_intent, average = 'macro', zero_division = 0)

    y_true_escalate = df_eval['gold_should_escalate'].astype(bool)
    escalate_p, escalate_r, escalate_f1, _ = precision_recall_fscore_support(y_true_escalate, preds_escalate, average = 'macro', zero_division = 0)

    average_rouge = sum(rouge_scores) / len(rouge_scores)

    return {
        "Model": model_name,
        "Intent Acc": f"{intent_acc*100:.1f}%",
        "Intent Macro F1": f"{intent_f1*100:.1f}%",
        "Escalation Recall": f"{escalate_r*100:.1f}%",
        "Escalation Precision": f"{escalate_p*100:.1f}%",
        "Escalation F1": f"{escalate_f1*100:.1f}%",
        "ROUGE-L": f"{average_rouge:.3f}"
    }

results = []
results.append(evaluate_model(trivial_model, df_test, 'Trivial Baseline'))
results.append(evaluate_model(simple_model, df_test, 'Simple Baseline (TF-IDF + Regex)'))
results.append(evaluate_model(support_agent, df_test, "Llama 3.1: 8B Parameters agent"))


df_results = pd.DataFrame(results)
df_results = pd.DataFrame(results)
print("\n" + "=" * 95)
print("                    FINAL HEADLINE BENCHMARK COMPARISON TABLE")
print("=" * 95)
print(df_results.to_string(index=False))
print("=" * 95)

df_results.to_csv("data/processed/headline_results.csv", index=False)
print("\n[✔] Results exported to 'data/processed/headline_results.csv'.")