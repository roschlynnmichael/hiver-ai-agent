# Report for AI Support Agent Project

**Candidate:** SDE Intern Application
**Target Brand:** `@AmazonHelp` (Twitter/X Customer Support)
**Core-Model:** Meta Llama 3.1 (8 Billion Parameters) via Local Ollama API + `all-MiniLM-L6-v2` for Vector RAG

## 1. Problem Framing & Non-Goals

### What 'good' means for @AmazonHelp?
Customer support on public social media differs greatly from internal helpdesks.
1. **Safety over Automation**: A false negative on escalation of queries destroys customer trust in the brand and can invite legal trouble. A false positive on the other hand just wastes slight human review time. 
2. **Strict PII confidentiality**: Agents should not request a customer's PII over social media in tweets that can causing doxxing or other issues for them. A good agent must redirect queries that require the customer to share any PII over to a human in private DMs or on Amazon's website.
3. **No fake promises or hallucinating concepts**: A good agent should never invent financial promises or concepts like "free refunds" or "guaranteed delivery dates" or "free returns". It must ground its replies in accordance with company defined policies and rubrics.

### What did I choose not to build?
1. **Automatically performing financial writes or executing tools**: I have avoided building direct API execution for refunds or address changes. Triaging customer queries is automatic while any financial decisions remain strictly upon a human to decide.
2. **Multi-Language Translation Pipeline**: Current version of the agent only converses in English and not multilingual.
3. **End to End Integration**: The agent is not deployed in a production environment. It is a command-line application that can be run locally.

### Results vs Baseline
All three systems were evaluated using unseen test-data drawn from the golden evaluation set using 75/25 train test split ensuring no data leakage.
| Model | Intent Accuracy | Intent Macro F1 | Escalation Recall | Escalation Precision | Escalation F1 | Reply ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** (Majority Intent + Never Escalate) | 25.5% | 6.8% | 50.0% | 37.2% | 42.7% | 0.118 |
| **Simple Baseline** (TF-IDF + Regex + 1-NN) | 46.8% | 34.4% | 61.1% | 77.0% | 62.3% | 0.112 |
| **Our Support Agent** (Llama 3.1 8B + RAG) | **83.0%** | **81.7%** | **90.2%** | **92.7%** | **91.4%** | **0.227** |

### Benchmark Analysis
1. Trivial Model achieved an accuracy of 25.5% and 6.8% Macro F1 because it blindly guessed `ORDER_TRACKING_DELAY` for every tweet scoring 0% across the other classes
2. The simple baseline model using TF Vectorizer achieved 46.8% accuracy. While its F1 was better it failed to generalize well to the test set  
3. Our Support Agent using Llama 3.1 8B LLM achieved an accuracy of 83.0% and 81.7% Macro F1 demonstrating the power of LLMs in understanding nuance in human language. It was also able to generalize well to unseen data.
4. Simple baseline model with its regex achieved a 61.8% escalation recall. Our agent achieved 90.2% recall alongside 92.7% precision. 
5. The trivial baseline model and simple baseline model with 1-NN tweet copy achieved only 0.11 Rouge score. My agent doubled that to 0.227 proving RAG Retrieval sucessfully injects authentic amazon resolution phrasing into generated replies.

## Failure Analysis
### Failure Mode 1: Sarcasm and Irony Inversion
* **Real Customer Tweet**: *"Thanks Amazon for delivering my TV in 50 pieces! Wonderful service as always! 👏"*
* **Agent Behavior**: Predicted `OUT_OF_SCOPE_CHITCHAT` or `ORDER_TRACKING_DELAY` with `should_escalate = False`.
* **Hypothesis**: The model's attention mechanism was biased by the positive surface tokens (*"Thanks"*, *"Wonderful"*, *"Great"*) and failed to detect the pragmatic sentiment inversion caused by *"in 50 pieces"*.

### Failure Mode 2: Multi-Intent Query Collapsing
* **Real Customer Tweet**: *"My package is 4 days late AND you charged my credit card twice! Cancel my Prime membership now."*
* **Agent Behavior**: Predicted `ORDER_TRACKING_DELAY` and addressed the delay, completely ignoring the double charge and cancellation demand.
* **Hypothesis**: Enforcing a strict single-label schema forced the model to collapse a multi-intent compound sentence into whichever clause appeared first, dropping the secondary financial risk.

### Failure Mode 3: Over-Escalation on Routine Impatience
* **Real Customer Tweet**: *"Ugh, tracking says delivery by 8 PM. Why does it always take so long? #annoyed"*
* **Agent Behavior**: Flagged `should_escalate = True` with reason *"Customer expresses severe anger"*.
* **Hypothesis**: The prompt's emphasis on high escalation recall caused the LLM to lower its threshold excessively, classifying mild, routine customer venting as actionable hostility.**

### Failure Mode 4: Negative Verification Blindness in LLM-as-a-Judge
* **Real Customer Tweet**: *"Where is my order? It was supposed to be delivered yesterday."*
* **Agent Reply**: *"I'm sorry for the delay! Can you please DM us with your order number so we can look into this further? ^EA"*
* **Judge Critique**: Scored **3/5**, stating: *"The response is polite and empathetic, but fails to redirect the customer to DM, which is a policy safety concern."*
* **Hypothesis**: The judge LLM suffered from hallucinated omission criteria—it generated a critique criticizing the absence of a DM directive despite the reply explicitly stating *"Can you please DM us"*.

### Failure Mode 5: Redundant Redirection ("Already in DM")
* **Real Customer Tweet**: *"I already sent 3 DMs with my order number and no one is replying! Answer me here!"*
* **Agent Reply**: *"Hi! We'd be happy to help. Please send us a DM with your order details."*
* **Hypothesis**: Without multi-turn dialogue context, the single-turn RAG retrieval injected standard resolution templates instructing the customer to DM, producing an infuriating, circular experience for a customer already waiting in DMs.

### What is misleading about my headline number?
During initial baseline development, our 1-Nearest Neighbor (1-NN) model scored **98.4% Intent Accuracy and 0.990 ROUGE-L**. This was a classic **train-on-test leakage artifact**: because 1-NN was fitted on the evaluation pool, it matched queries to themselves with cosine similarity `1.0` and copied the target ground-truth reply verbatim. Once split into unseen train/test partitions, the Simple Baseline's true ROUGE-L collapsed to **0.112**. Any benchmark claiming >0.70 ROUGE on open-ended customer support replies is almost certainly suffering from evaluation leakage.

If given one more week, here's what I would do to enhance it:
1. Fine tune a lightweight 3 Billion Parameter model to enable faster response and analysis.
2. Add multiple labels to customer queries
3. Ingest prior 3 customer and brand tweets to provide better context to the LLM.
4. Automatically hide PII information of customers who accidently add them to tweets.

### Why I decided to do the task this way?
1. I choose `@AmazonHelp` due to availability of broader oeprational risks compared to airline related queries.
2. Chunked streaming to avoid memory kernel panics on consumer hardware.
3. Stripping of URLs and Screenshots from tweets using Regex
4. Chose locally served Ollama models to avoid API rate limits and ensure complete data privacy.
5. Ran embeddings on CPU rather than Ollama embeddings to avoid ollama from swapping models and increasing load times.
6. Preventing data leakages of 1-NN model by following a 75% training data and 25% unseen testing data approach.
7. Included a "simple baseline" and "trivial baseline" to establish fair comparison and to show the utility of the LLM.
8. Used "LLM-as-a-Judge" for evaluation to better approximate human judgment on open-ended generation tasks.
9. Set `temperature = 0.0` for proper triage classification and escalation decisions.
10. Merged intent classification and escalation decision into a single API call to reduce LLM inference latency by 50%