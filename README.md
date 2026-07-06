# Line-by-Line Explanation: `rag_negation_v2.ipynb`

**Research question:** Does stripping negation words (like "not") during retrieval cause a RAG pipeline to fetch the wrong passage — and does that wrong passage then cause the downstream model to give the wrong final answer, *compounding* the two errors?

The notebook has 3 experiments: (1) retrieval-only test on NevIR, (2) fine-tuning DistilBERT on CondaQA, (3) chaining the two together to see if errors compound. Below is every code cell explained line by line, with what output to expect.

---

## The Problem, In Plain English

Before diving into code, here's the plain-language version of what this notebook is actually testing.

**What is a RAG system?**
A lot of AI chatbots (customer support bots, legal/medical assistants, "chat with your documents" tools) work in two steps:
1. **Retriever** — a search engine that looks through a pile of documents and pulls out the one that seems most relevant to your question.
2. **Generator** — a language model (like DistilBERT or GPT) that reads whatever document the retriever handed it and writes an answer.

The generator never sees the whole document library — it only ever sees what the retriever chose to show it. If the retriever hands it the wrong document, the generator has no way of knowing that; it just answers based on what's in front of it.

**Where negation breaks this**
Compare these two sentences:
- "This medication is effective for treating infections."
- "This medication is **not** effective for treating infections."

They mean the exact opposite. But a lot of retrievers — especially the classic keyword-based kind (BM25) — were built in an era when common short words like "the," "is," "a," and **"not"** were treated as "stopwords": noise words to be deleted before search, because they were assumed to carry no useful meaning. So a typical old-school cleaning step turns both sentences above into the exact same bag of words:

```
medication effective treating infections
```

Once "not" is deleted, the retriever literally cannot tell the two documents apart anymore — they look identical to it. Ask it "is this medication effective?" and it might hand back the negative document as if it were the positive one, with total confidence, because from its point of view they're the same document.

**Why this is a bigger problem now than it used to be**
Modern language models (the generator half) are actually very good at understanding negation — they're trained on huge amounts of real text and pick up that "not effective" means the opposite of "effective." But that skill only helps if the generator is given the correct document in the first place. The retriever half of the system never got that same upgrade — a lot of production systems still use 1990s-style keyword search with the same old stopword lists. So you end up with a smart reader being fed the wrong page.

**The actual question this notebook is trying to answer**
People have separately studied "can retrievers handle negation?" and "can language models handle negation?" — but rarely both *chained together*. That chaining matters because errors might not just add up — they might make each other worse ("compound"). For example:
- If the retriever is wrong 40% of the time, and the model (given the right document) is wrong 20% of the time, naive math says the combined system should be right about as often as `(1 - 0.40) × (1 - 0.20) ≈ 48%` of the time.
- But if the model doesn't notice when it's been handed the wrong document — it just answers confidently anyway — the real combined accuracy could be much worse than that naive estimate.

That gap between "predicted accuracy if errors were independent" and "actual accuracy of the real chained system" is the core thing this notebook measures.

**How the notebook tests this, one piece at a time**
- **Experiment 1** — isolate the retriever. Give it two nearly-identical documents differing only by the word "not," and see how often it picks the right one.
- **Experiment 2** — isolate the generator. Skip retrieval entirely, always give the model the correct document, and see how well it answers negation-based yes/no questions (before and after fine-tuning).
- **Experiment 3** — chain them together for real, and compare the actual accuracy against the "if errors were independent" prediction from Experiments 1 and 2.

**One-sentence summary:** classic retrieval systems delete negation words like "not" before searching, which makes them confuse opposite-meaning documents — and because the smart language model downstream just trusts and answers whatever document it's handed, the whole system's accuracy ends up worse than you'd predict from each part's error rate alone.

---

## Section 0 — Setup

### Cell: `!pip -q install ...`
```
!pip -q install datasets rank_bm25 sentence-transformers transformers \
     accelerate matplotlib seaborn pandas scikit-learn --upgrade
```
- `!pip install` runs a shell pip install from inside the notebook (the `!` prefix means "run this as a shell command, not Python").
- `-q` = quiet mode, suppresses most install logs.
- Installs: `datasets` (HuggingFace dataset loader), `rank_bm25` (BM25 keyword search), `sentence-transformers` (dense embeddings), `transformers` (DistilBERT), `accelerate` (HF training speedups), `matplotlib`/`seaborn` (plots), `pandas` (dataframes), `scikit-learn` (metrics).
- **Expected output:** a stream of "Successfully installed ..." lines, or nothing visible if already installed (because of `-q`). No errors expected. This can take 1–3 minutes on Colab.

### Cell: imports and global constants
```python
import random, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, f1_score)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook", palette="muted")
RNG = np.random.default_rng(42)
random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

NEVIR_SAMPLE  = 200
TRAIN_SIZE    = 500
VAL_SIZE      = 150
TEST_SIZE     = 200
EPOCHS        = 4
BATCH_SIZE    = 16
MAX_LEN       = 256
LR            = 2e-5
MODEL_NAME    = "distilbert-base-uncased"
```
- Standard library imports: `random` (RNG), `re` (regex, used later to strip punctuation), `warnings` (to silence noisy library warnings).
- `numpy`/`pandas`: array math and dataframes.
- `matplotlib`/`seaborn`: plotting.
- `Counter` is imported but not actually used anywhere later in the notebook (harmless).
- `torch` and friends: PyTorch, plus `Dataset`/`DataLoader` for the custom training pipeline, `AutoTokenizer`/`AutoModelForSequenceClassification` to load DistilBERT, `get_linear_schedule_with_warmup` for the LR schedule, `AdamW` optimizer.
- `load_dataset` pulls datasets from the HuggingFace Hub.
- `BM25Okapi` is the BM25 keyword-ranking algorithm.
- `SentenceTransformer` loads a dense embedding model.
- `sklearn.metrics` items are used for evaluation later (`ConfusionMatrixDisplay` is imported but never actually called — `sns.heatmap` is used instead).
- `warnings.filterwarnings("ignore")` hides deprecation/future warnings so output stays clean.
- `sns.set_theme(...)` sets a consistent plot style (white grid background, "muted" color palette) for every chart in the notebook.
- Three separate seeds are set (`np.random.default_rng(42)`, `random.seed(42)`, `torch.manual_seed(42)`) so sampling and training are reproducible — note `RNG` (numpy generator) is created but never actually used later; the notebook uses `random_state=42` directly in pandas `.sample()` calls instead.
- `DEVICE` picks GPU if available, otherwise CPU. **Expected output:** `Device: cuda` on a Colab GPU runtime, or `Device: cpu` otherwise.
- The constants block defines how much data to sample and hyperparameters: 200 NevIR examples, 500 training / 150 validation / 200 test CondaQA examples, 4 training epochs, batch size 16, max token length 256, learning rate 2e-5, and the model `distilbert-base-uncased`.

---

## Section 1 — Load Datasets

### Cell: load NevIR
```python
nevir_raw = load_dataset("orionweller/NevIR")
nevir_split = "test" if "test" in nevir_raw else list(nevir_raw.keys())[0]
nevir_full = pd.DataFrame(nevir_raw[nevir_split])
print("NevIR shape:", nevir_full.shape)
print("NevIR columns:", list(nevir_full.columns))
nevir_full.head(2)
```
- Downloads the NevIR dataset (negation-contrastive retrieval pairs) from the HuggingFace Hub. This returns a `DatasetDict` with splits like `train`/`validation`/`test`.
- `nevir_split` picks the `"test"` split if it exists, otherwise falls back to whatever the first available split key is — defensive coding in case the dataset's split names differ from expectations.
- Converts that split into a pandas DataFrame for easier manipulation.
- **Expected output:** a shape tuple like `NevIR shape: (250, 4)` (exact row count/columns depend on the dataset version), a column list such as `['q1', 'q2', 'doc1', 'doc2']` (or similarly named), and a rendered table showing the first 2 rows.

### Cell: load CondaQA
```python
condaqa_raw = load_dataset("lasha-nlp/CONDAQA")
print("CondaQA splits:", list(condaqa_raw.keys()))
print("CondaQA columns:", list(condaqa_raw[list(condaqa_raw.keys())[0]].column_names))
```
- Downloads CondaQA (negation-contrastive QA dataset).
- Prints the split names (**expected:** something like `['train', 'dev', 'test']`).
- Prints the column names of the first split, so you can confirm the schema before hardcoding column names in the next cell (**expected:** a list including columns like `original passage`, `sentence1`, `sentence2`, `label`, `original cue`, etc.).

### Cell: define column mapping and build CondaQA yes/no splits
```python
PASSAGE_COL  = "original passage"
QUESTION_COL = "sentence1"
ANSWER_COL   = "label"
NEGCUE_COL   = "original cue"
EDITED_COL   = "sentence2"

def load_condaqa_split(split_name):
    df = pd.DataFrame(condaqa_raw[split_name])
    df["answer_clean"] = df[ANSWER_COL].astype(str).str.strip().str.lower()
    label_map = {"1":"yes","0":"no","yes":"yes","no":"no","true":"yes","false":"no"}
    df["answer_clean"] = df["answer_clean"].map(label_map)
    df = df[df["answer_clean"].isin(["yes","no"])].copy()
    df["label_int"] = (df["answer_clean"] == "yes").astype(int)
    return df.reset_index(drop=True)

train_df_full = load_condaqa_split("train")
dev_df_full   = load_condaqa_split("dev")
test_df_full  = load_condaqa_split("test")

print(f"CondaQA  train (yes/no): {len(train_df_full)}")
print(f"CondaQA  dev   (yes/no): {len(dev_df_full)}")
print(f"CondaQA  test  (yes/no): {len(test_df_full)}")
test_df_full[[PASSAGE_COL, QUESTION_COL, "answer_clean"]].head(3)
```
- Five constants hardcode which raw CondaQA columns to use: the passage text, the question (oddly named `sentence1` in the raw data), the raw answer label, the negation-cue word, and the edited/counterfactual sentence (`sentence2`).
- `load_condaqa_split(split_name)`:
  - Converts a split to a DataFrame.
  - `answer_clean` lowercases/strips the raw label.
  - `label_map` normalizes many possible label spellings (`"1"`, `"true"`, `"yes"` → `"yes"`; `"0"`, `"false"`, `"no"` → `"no"`) into a consistent `"yes"`/`"no"` string.
  - Filters out any row whose label didn't map cleanly to yes/no (CondaQA has some non-boolean answer types like extractive spans, which get dropped here since this notebook only wants a binary classification task).
  - `label_int` converts `"yes"`→`1`, `"no"`→`0` for model training.
  - Resets the row index after filtering.
- Loads train/dev/test with this function and prints counts.
- **Expected output:** three counts (e.g., `CondaQA train (yes/no): 3400ish`, etc. — exact numbers depend on how many yes/no rows exist after filtering), followed by a small table of 3 example rows showing the passage, question, and clean yes/no answer.

---

## Section 2 — Exploratory Data Analysis (EDA)

### Cell: normalize NevIR column names
```python
def pick(df, opts):
    for o in opts:
        if o in df.columns: return o
    raise KeyError(f"None of {opts} in {list(df.columns)}")

nevir_full = nevir_full.rename(columns={
    pick(nevir_full, ["q1","query1","question1"]): "q1",
    pick(nevir_full, ["q2","query2","question2"]): "q2",
    pick(nevir_full, ["doc1","document1","passage1"]): "doc1",
    pick(nevir_full, ["doc2","document2","passage2"]): "doc2",
})
nevir_full[["q1","q2","doc1","doc2"]].head(3)
```
- `pick()` is a small helper that tries several possible column-name spellings and returns whichever one actually exists — a defensive pattern since dataset column names can vary by version.
- Renames whatever the real column names are to the canonical `q1`, `q2`, `doc1`, `doc2` so the rest of the notebook can refer to them consistently.
- **Expected output:** a table with 3 rows showing query 1, query 2, document 1, document 2 — each pair differing only by a negation word (e.g., "the treatment was effective" vs "the treatment was **not** effective").

### Cell: 2.1 NevIR length distributions
```python
nevir_full["q1_len"]   = nevir_full["q1"].str.split().apply(len)
nevir_full["q2_len"]   = nevir_full["q2"].str.split().apply(len)
nevir_full["doc1_len"] = nevir_full["doc1"].str.split().apply(len)
nevir_full["doc2_len"] = nevir_full["doc2"].str.split().apply(len)
nevir_full["len_diff"] = (nevir_full["doc1_len"] - nevir_full["doc2_len"]).abs()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# ... (3 histograms) ...
plt.tight_layout()
plt.savefig("nevir_length_distributions.png", dpi=120)
plt.show()
print(f"Median query length:  {q_lens.median():.0f} words")
print(f"Median doc length:    {d_lens.median():.0f} words")
print(f"Median intra-pair diff: {nevir_full['len_diff'].median():.1f} words")
print("A small intra-pair diff confirms these are *minimal* edits — the")
print("two docs in a pair are nearly identical except for the negation.")
```
- Computes word counts for each query and document by splitting on whitespace and counting tokens.
- `len_diff` = absolute difference in word count between doc1 and doc2 within a pair — a way to check that the two documents are near-identical (differing mainly by the negation word), not by length.
- Builds a 3-panel figure:
  1. Histogram of query word lengths (q1 and q2 combined).
  2. Histogram of document word lengths (doc1 and doc2 combined).
  3. Histogram of the intra-pair length difference.
- Saves the figure to `nevir_length_distributions.png` and displays it inline.
- **Expected output:** three histograms rendered inline, plus three printed lines reporting median query length, median doc length, and median length difference (this last number should be small — e.g., 0–2 words — confirming the pairs are near-identical except for the negation).

### Cell: 2.2 CondaQA label balance
```python
all_cqa = pd.concat([train_df_full, dev_df_full, test_df_full])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = all_cqa["answer_clean"].value_counts()
axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", ...)
# per-split bar chart
split_counts = pd.DataFrame({...}).T
split_counts.plot(kind="bar", ax=axes[1], ...)
plt.savefig("condaqa_label_distribution.png", dpi=120)
plt.show()
print("Insight: roughly balanced labels mean accuracy is a fair metric.")
```
- Concatenates all three CondaQA splits into one DataFrame for a combined view.
- Left panel: pie chart of overall "yes" vs "no" label proportions.
- Right panel: grouped bar chart of yes/no counts broken out by split (train/dev/test).
- **Expected output:** a pie chart (should be close to 50/50 if the dataset is balanced) and a bar chart, plus the printed insight line. If labels are close to 50/50, plain accuracy is a trustworthy metric later (no need for weighted metrics).

### Cell: 2.3 CondaQA passage length distribution
```python
all_cqa["passage_len"] = all_cqa[PASSAGE_COL].astype(str).str.split().apply(len)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(all_cqa["passage_len"], ...)   # left: histogram
sns.boxplot(data=..., x="split", y="words", ...)  # right: box plot per split
plt.savefig("condaqa_passage_lengths.png", dpi=120)
plt.show()
```
- Computes passage word count.
- Left: histogram of passage lengths across all data.
- Right: box plot comparing passage-length distributions per split (train/dev/test), built by melting a dict of three per-split length series into long format for seaborn.
- **Expected output:** a histogram (likely right-skewed, some passages much longer than others) and a box plot with three boxes (train/dev/test) that should look similar to each other if the splits are drawn from the same distribution.

### Cell: 2.4 Most common negation cues
```python
cue_counts = all_cqa[NEGCUE_COL].astype(str).str.strip().str.lower().value_counts().head(20)
plt.figure(figsize=(9, 5))
sns.barplot(x=cue_counts.values, y=cue_counts.index, color="#378ADD")
plt.title("CondaQA: top 20 negation cues")
plt.savefig("condaqa_negation_cues.png", dpi=120)
plt.show()
print("These are the actual negation words in the passages.")
print("Removing these as 'stopwords' is exactly what breaks model understanding.")
```
- Cleans and counts the `original cue` column (the specific negation word/phrase each example is built around), takes the top 20.
- Horizontal bar chart of the top 20 negation cues by frequency.
- **Expected output:** a bar chart with words like "not", "no", "never", "n't" etc. dominating, plus the two printed sentences — this sets up the notebook's core thesis (that stripping these as stopwords is harmful).

### Cell: 2.5 NevIR word-overlap between pairs
```python
from difflib import SequenceMatcher

def overlap(a, b):
    return SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()

sample_for_overlap = nevir_full.sample(n=min(100, len(nevir_full)), random_state=42)
sample_for_overlap["overlap"] = sample_for_overlap.apply(
    lambda r: overlap(r["doc1"], r["doc2"]), axis=1)

plt.figure(figsize=(7, 4))
sns.histplot(sample_for_overlap["overlap"], bins=20, color="#FF8C00")
plt.axvline(sample_for_overlap["overlap"].median(), ...)
plt.savefig("nevir_pair_overlap.png", dpi=120)
plt.show()
print("High overlap (near 1.0) confirms the two docs in a pair are nearly")
print("identical — the only real difference is the negation word.")
```
- `SequenceMatcher(...).ratio()` computes a similarity score (0 to 1) between the word-token sequences of doc1 and doc2 — essentially a "how similar are these two lists of words" score, higher = more similar.
- Samples up to 100 pairs (fixed seed for reproducibility) and computes this overlap ratio for each pair.
- Plots a histogram of overlap ratios with a red dashed line at the median.
- **Expected output:** histogram concentrated near 0.8–1.0 (very high overlap), confirming doc1/doc2 pairs are near-duplicates apart from the negation.

---

## Section 3 — Experiment 1: Retrieval-Only Test (NevIR)

### Cell: BM25 retrieval with stopword-stripped cleaning
```python
STOPWORDS = set((
    "a an the of to in on for and or is are was were be been being "
    "this that these those it its as at by with from not no nor so but if "
    "then than i you he she we they his her their our your have has had "
    "do does did will would could should may might must shall"
).split())

def clean_bm25(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

def bm25_pick(query_tokens, docs_tokens):
    bm25 = BM25Okapi(docs_tokens)
    scores = bm25.get_scores(query_tokens)
    return int(np.argmax(scores))

nevir_sample = nevir_full.sample(
    n=min(NEVIR_SAMPLE, len(nevir_full)), random_state=42).reset_index(drop=True)

bm25_rows = []
for _, row in nevir_sample.iterrows():
    d1_tok = clean_bm25(row["doc1"])
    d2_tok = clean_bm25(row["doc2"])
    p1 = bm25_pick(clean_bm25(row["q1"]), [d1_tok, d2_tok])
    p2 = bm25_pick(clean_bm25(row["q2"]), [d1_tok, d2_tok])
    bm25_rows.append({
        "q1_correct": p1 == 0,
        "q2_correct": p2 == 1,
        "pair_correct": (p1 == 0) and (p2 == 1),
    })

bm25_df = pd.DataFrame(bm25_rows)
bm25_single = bm25_df[["q1_correct","q2_correct"]].values.mean()
bm25_pair   = bm25_df["pair_correct"].mean()
print(f"BM25 single-query accuracy : {bm25_single:.3f}")
print(f"BM25 strict pairwise       : {bm25_pair:.3f}")
```
- **This is the crux of the "filing mistake" hypothesis.** `STOPWORDS` is a hand-written stoplist that critically **includes "not" and "no"** — mimicking a classic (flawed) text-preprocessing pipeline.
- `clean_bm25(text)`: lowercases, strips all non-alphanumeric characters via regex, splits into tokens, and drops any token that's in the stopword list or a single character. Because "not"/"no" are in `STOPWORDS`, negation signal is destroyed here.
- `bm25_pick(query_tokens, docs_tokens)`: builds a fresh BM25 index over the (2-document) corpus for this pair, scores the query against both documents, and returns the index (0 or 1) of the highest-scoring document.
- Samples 200 NevIR pairs (or fewer if the dataset is smaller) with a fixed seed for reproducibility.
- For each pair: cleans both docs, then asks BM25 "given query 1, is doc1 or doc2 the better match?" (correct answer is doc1, i.e., index 0) and "given query 2, is doc1 or doc2 better?" (correct answer is doc2, i.e., index 1).
- Records three booleans per pair: whether q1 alone was correct, whether q2 alone was correct, and whether **both** were correct (`pair_correct`, the strict pairwise metric NevIR actually uses).
- `bm25_single` = average correctness across all individual queries (400 queries total if sample=200). `bm25_pair` = fraction of pairs where both queries were answered correctly.
- **Expected output:** Two printed accuracy numbers. Because negation words are stripped, doc1 and doc2 become nearly identical after cleaning, so scores for both docs are similar — expect BM25 to perform close to random: single-query accuracy near 0.5, and pairwise accuracy near 0.25 (or lower), since guessing right on both queries by chance is much harder.

### Cell: Dense retrieval with MiniLM (no cleaning)
```python
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def dense_pick(query, doc1, doc2):
    q  = embed_model.encode(query, convert_to_tensor=True)
    ds = embed_model.encode([doc1, doc2], convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(q.unsqueeze(0), ds)
    return int(torch.argmax(sims).item())

dense_rows = []
for _, row in nevir_sample.iterrows():
    p1 = dense_pick(row["q1"], row["doc1"], row["doc2"])
    p2 = dense_pick(row["q2"], row["doc1"], row["doc2"])
    dense_rows.append({...})

dense_df = pd.DataFrame(dense_rows)
dense_single = dense_df[["q1_correct","q2_correct"]].values.mean()
dense_pair   = dense_df["pair_correct"].mean()
print(f"Dense single-query accuracy: {dense_single:.3f}")
print(f"Dense strict pairwise      : {dense_pair:.3f}")
```
- Loads `all-MiniLM-L6-v2`, a small pretrained sentence-embedding model (downloads weights the first time this runs — may take a bit).
- `dense_pick`: embeds the query and both raw (uncleaned) documents into vectors, computes cosine similarity between the query vector and each document vector, and returns the index of the more similar document. Unlike BM25 here, **no stopword stripping is applied** — the model sees "not" in context.
- Runs the same per-pair, per-query loop as the BM25 experiment, producing the same three metrics.
- **Expected output:** Two printed accuracy numbers, likely somewhat better than BM25's but still well below what you'd want for a negation-sensitive task — general-purpose sentence embeddings are known to struggle with negation because "X is effective" and "X is not effective" are semantically/lexically very similar in embedding space. Expect single-query accuracy noticeably above 0.5 but pairwise accuracy still fairly low.

### Cell: Retrieval comparison chart
```python
retr = pd.DataFrame({
    "Retriever": ["Random baseline","BM25 (cleaned)","Dense MiniLM (raw)"],
    "Single-query accuracy": [0.50, bm25_single, dense_single],
    "Strict pairwise accuracy": [0.25, bm25_pair, dense_pair],
})
print(retr.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# left: grouped bar of single vs pairwise accuracy per retriever, with random-baseline lines
# right: pie chart breaking BM25 pairwise outcomes into 4 categories
plt.savefig("experiment1_retrieval.png", dpi=120)
plt.show()
```
- Builds a small summary table comparing Random (theoretical 0.5/0.25), BM25, and Dense retrieval side by side and prints it as plain text.
- Left subplot: for each retriever, plots a solid bar for single-query accuracy and a lighter/semi-transparent bar for pairwise accuracy, with dashed reference lines at 0.5 and 0.25 (random-guess levels).
- Right subplot: pie chart breaking BM25's outcomes into 4 buckets — both queries correct, only Q1 correct, only Q2 correct, both wrong.
- **Expected output:** a printed table, then a 2-panel figure — left showing all three retrievers' bars, right showing the BM25 error breakdown pie (expect "both wrong" and "only one correct" slices to be large if BM25 performs near chance).

### Cell: Per-pair error analysis (length/overlap vs correctness)
```python
nevir_sample["bm25_pair_correct"] = bm25_df["pair_correct"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# left: KDE of document length, split by correct/wrong
for correct, grp in nevir_sample.groupby("bm25_pair_correct"):
    ...
    sns.kdeplot(d_len, ax=axes[0], label=label, color=col, fill=True, alpha=0.3)
# right: scatter of q1_len vs q2_len colored by correctness
axes[1].scatter(nevir_sample["q1_len"], nevir_sample["q2_len"], c=..., alpha=0.5, s=20)
plt.savefig("experiment1_error_analysis.png", dpi=120)
plt.show()
```
- Attaches the BM25 pair-correctness flag back onto the sampled NevIR dataframe.
- Left: overlapping density plots (KDE) of document length, one curve for pairs BM25 got right, one for pairs it got wrong — checks whether errors correlate with document length.
- Right: scatter plot of query-1 length vs query-2 length, colored green (correct) or red (wrong) per pair, with a manual legend built via `matplotlib.patches.Patch`.
- **Expected output:** two plots. If there's no strong relationship between length and correctness (which is the likely finding, since the failure mode is about negation-word removal, not length), the correct/wrong distributions should look fairly similar/overlapping rather than cleanly separated.

---

## Section 4 — Experiment 2: Fine-Tune DistilBERT on CondaQA

### Cell: Sample splits and build PyTorch Dataset
```python
train_df = train_df_full.sample(n=min(TRAIN_SIZE, len(train_df_full)), random_state=42)
val_df   = dev_df_full.sample(n=min(VAL_SIZE,   len(dev_df_full)),   random_state=42)
test_df  = test_df_full.sample(n=min(TEST_SIZE,  len(test_df_full)),  random_state=42)

print(f"Fine-tune train : {len(train_df)}")
print(f"Validation      : {len(val_df)}")
print(f"Test            : {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CondaQADataset(Dataset):
    def __init__(self, df):
        self.questions = df[QUESTION_COL].astype(str).tolist()
        self.passages  = df[PASSAGE_COL].astype(str).tolist()
        self.labels    = df["label_int"].tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.questions[idx], self.passages[idx],
            truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[idx])

train_ds = CondaQADataset(train_df)
val_ds   = CondaQADataset(val_df)
test_ds  = CondaQADataset(test_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)
print("Data loaders ready.")
```
- Randomly subsamples 500/150/200 rows (or fewer if not that many exist) from the full train/dev/test splits, using the sizes set in Section 0.
- Loads the DistilBERT tokenizer.
- `CondaQADataset` is a custom PyTorch `Dataset`:
  - Stores questions, passages, and integer labels as plain lists.
  - `__getitem__` tokenizes a **question+passage pair** together (DistilBERT sees `[CLS] question [SEP] passage [SEP]`), truncating/padding to `MAX_LEN=256` tokens, and returns a dict of tensors (`input_ids`, `attention_mask`, etc.) plus the label as a tensor. `.squeeze(0)` removes the batch dimension that the tokenizer adds by default so each item is a single, unbatched example (the `DataLoader` will re-batch them later).
- Wraps train/val/test datasets in `DataLoader`s; training data is shuffled each epoch, val/test are not.
- **Expected output:** three printed counts (500, 150, 200 unless the underlying splits are smaller), then `Data loaders ready.` The tokenizer download happens silently the first time.

### Cell: Zero-shot baseline
```python
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds = torch.argmax(out.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_labels, all_preds

zeroshot_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2).to(DEVICE)
zs_acc, zs_f1, _, _ = evaluate(zeroshot_model, test_loader, DEVICE)
print(f"Zero-shot accuracy : {zs_acc:.3f}")
print(f"Zero-shot macro-F1 : {zs_f1:.3f}")
```
- `evaluate()` is a reusable function: puts the model in eval mode, disables gradient tracking (`torch.no_grad()`), loops over a data loader, computes predictions via `argmax` over the 2-class logits, and returns accuracy plus macro-F1 (F1 averaged equally across both classes, robust to class imbalance) along with the raw labels/predictions for later confusion-matrix plotting.
- Loads a **fresh, untrained** classification head on top of pretrained DistilBERT weights (`num_labels=2` adds a randomly-initialized linear layer for yes/no classification — the base transformer weights are pretrained, but the classification head is not trained yet).
- Evaluates this zero-shot model directly on the test set.
- **Expected output:** Because the classification head is randomly initialized and never trained, expect accuracy close to random chance — **around 0.50**, and a low/unstable macro-F1 (possibly the model predicts one class for almost everything, which would tank F1 for the minority-predicted class even if overall accuracy looks ~50%).

### Cell: Fine-tuning loop
```python
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler  = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    ep_loss, ep_correct, ep_total = 0.0, 0, 0
    for batch, labels in train_loader:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(**batch)
        loss = criterion(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        ep_loss    += loss.item() * labels.size(0)
        ep_correct += (out.logits.argmax(1) == labels).sum().item()
        ep_total   += labels.size(0)

    val_acc, val_f1, _, _ = evaluate(model, val_loader, DEVICE)

    model.eval(); v_loss = 0.0; v_total = 0
    with torch.no_grad():
        for batch, labels in val_loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = labels.to(DEVICE)
            v_loss  += criterion(model(**batch).logits, labels).item() * labels.size(0)
            v_total += labels.size(0)

    history["train_loss"].append(ep_loss / ep_total)
    history["train_acc"].append(ep_correct / ep_total)
    history["val_loss"].append(v_loss / v_total)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"train_loss={ep_loss/ep_total:.4f}  train_acc={ep_correct/ep_total:.3f}  "
          f"val_loss={v_loss/v_total:.4f}  val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

print("Training complete.")
```
- Loads a **second, separate** fresh DistilBERT model (`model`, distinct from `zeroshot_model`) — this is the one that will actually be trained.
- `AdamW` optimizer with the configured learning rate and 0.01 weight decay (L2 regularization to reduce overfitting).
- `total_steps` = number of batches per epoch × number of epochs — needed to configure the LR schedule.
- `get_linear_schedule_with_warmup`: learning rate linearly increases for the first 10% of steps (warmup), then linearly decays to 0 for the rest — a standard transformer fine-tuning schedule.
- `history` dict will store per-epoch metrics for later plotting.
- `criterion = CrossEntropyLoss()` — standard loss for multi-class (here binary) classification from raw logits.
- Training loop, once per epoch:
  - `model.train()` enables dropout etc.
  - For each batch: move data to device, zero gradients, forward pass, compute loss, backward pass, **gradient clipping** (`clip_grad_norm_(..., 1.0)` prevents exploding gradients by capping the total gradient norm at 1.0), optimizer + scheduler step.
  - Accumulates weighted loss and correct-prediction counts to compute per-epoch averages.
  - After each epoch, evaluates on the validation set via `evaluate()` for accuracy/F1, then **separately** recomputes validation loss (a bit redundant but done explicitly here since `evaluate()` doesn't return loss).
  - Appends all four metrics to `history` and prints an epoch summary line.
- **Expected output:** 4 lines like:
  ```
  Epoch 1/4  train_loss=0.65..  train_acc=0.6..  val_loss=0.55..  val_acc=0.7..  val_f1=0.7..
  Epoch 2/4  train_loss=0.4...  train_acc=0.8..  val_loss=0.45..  val_acc=0.75. val_f1=0.75.
  ...
  ```
  followed by `Training complete.` Exact numbers vary by run/data, but you should see train_loss decrease and train_acc increase epoch over epoch, with val_acc generally improving too (though it may plateau given the small `TRAIN_SIZE=500`).

### Cell: Training curves
```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ep_x = range(1, EPOCHS+1)
axes[0].plot(ep_x, history["train_loss"], "o-", ...)
axes[0].plot(ep_x, history["val_loss"],   "s--", ...)
axes[1].plot(ep_x, history["train_acc"], "o-", ...)
axes[1].plot(ep_x, history["val_acc"],   "s--", ...)
plt.savefig("training_curves.png", dpi=120)
plt.show()
print("If val_loss stops improving or rises while train_loss falls,")
print("that is a sign of overfitting. With EPOCHS=4 and a large dataset this")
print("is unlikely, but good to check.")
```
- Plots the 4 epochs of loss (left) and accuracy (right), with train as solid circles and val as dashed squares.
- **Expected output:** two line charts, each with 2 curves across 4 x-axis points (epochs 1–4), plus the printed caveat about overfitting. Given the small `TRAIN_SIZE=500`, some divergence between train and val curves is plausible and worth watching for.

### Cell: Evaluate fine-tuned model on test set
```python
ft_acc, ft_f1, true_labels, pred_labels = evaluate(model, test_loader, DEVICE)
print(f"Fine-tuned accuracy : {ft_acc:.3f}")
print(f"Fine-tuned macro-F1 : {ft_f1:.3f}")
print()
print(classification_report(true_labels, pred_labels, target_names=["No","Yes"]))
```
- Runs the now-trained `model` on the held-out test set.
- `classification_report` prints per-class precision, recall, F1, and support, plus overall accuracy/macro/weighted averages.
- **Expected output:** two summary numbers, likely meaningfully higher than the ~0.50 zero-shot baseline (fine-tuning should help substantially even with only 500 training examples — expect something in a 0.65–0.85 accuracy range depending on data/luck), followed by the full scikit-learn classification report table.

### Cell: Confusion matrices — zero-shot vs fine-tuned
```python
_, _, zs_true, zs_pred = evaluate(zeroshot_model, test_loader, DEVICE)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, t, p, title in [
        (axes[0], zs_true, zs_pred, f"Zero-shot DistilBERT\n(acc={zs_acc:.2f})"),
        (axes[1], true_labels, pred_labels, f"Fine-tuned DistilBERT\n(acc={ft_acc:.2f})")]:
    cm = confusion_matrix(t, p)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ...)
    ...
plt.savefig("confusion_matrices.png", dpi=120)
plt.show()
print("Reading this: the diagonal cells are correct predictions.")
print("The top-right cell (actual=No, predicted=Yes) is the model being tricked")
print("by negation — it sees a negated passage and answers Yes when it should say No.")
```
- Re-runs `evaluate()` on the zero-shot model to get its raw predictions again (needed here since they weren't saved earlier — this re-runs the forward pass a second time, a minor inefficiency but harmless).
- Draws two heatmap confusion matrices side by side: zero-shot vs fine-tuned, both on the same test set, both annotated with raw counts.
- **Expected output:** two 2×2 heatmaps. The zero-shot one should look close to random (roughly even counts in all 4 cells, or all predictions dumped into one column). The fine-tuned one should show much stronger diagonal concentration (more correct predictions), though the "actual=No, predicted=Yes" cell (top-right) may still show meaningful mass if negation confusion persists.

### Cell: Per-class accuracy bar chart
```python
def per_class_acc(true, pred, labels):
    return {labels[i]: np.mean([p == t for t, p in zip(true, pred) if t == i])
            for i in range(len(labels))}

zs_pca = per_class_acc(zs_true, zs_pred, ["No","Yes"])
ft_pca = per_class_acc(true_labels, pred_labels, ["No","Yes"])

pca_df = pd.DataFrame({"Zero-shot": zs_pca, "Fine-tuned": ft_pca})
ax = pca_df.plot(kind="bar", figsize=(7, 4), color=["#888780","#0F6E56"], rot=0)
plt.axhline(0.5, color="red", linestyle="--", lw=1, label="Random")
plt.savefig("per_class_accuracy.png", dpi=120)
plt.show()
print("Pay attention to asymmetry: if 'No' accuracy is much lower than 'Yes',")
print("the model is still confused by negation even after fine-tuning.")
```
- `per_class_acc`: for each true label value (0="No", 1="Yes"), computes the fraction of examples of that class the model got right — i.e., recall per class.
- Builds a small dataframe with rows "No"/"Yes" and columns "Zero-shot"/"Fine-tuned", plots as a grouped bar chart with a red dashed line at 0.5 (random-guess reference).
- **Expected output:** a bar chart with 2 groups (No, Yes) × 2 bars each (zero-shot, fine-tuned). Fine-tuned bars should generally be taller than zero-shot. Watch for asymmetry — if "No" stays low even after fine-tuning, that's direct evidence the model still struggles with negated ("No") passages specifically.

### Cell: Accuracy improvement summary
```python
improve_df = pd.DataFrame({
    "Model": ["Zero-shot DistilBERT", "Fine-tuned DistilBERT"],
    "Accuracy": [zs_acc, ft_acc],
    "Macro-F1":  [zs_f1, ft_f1],
})
display(improve_df)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
ax.bar(x - 0.2, improve_df["Accuracy"], 0.35, ...)
ax.bar(x + 0.2, improve_df["Macro-F1"],  0.35, ...)
ax.axhline(0.5, color="red", linestyle="--", lw=1, label="Random baseline")
plt.savefig("improvement_summary.png", dpi=120)
plt.show()
```
- `display(improve_df)` — a Jupyter/IPython-only function (works in Colab/Jupyter, would raise `NameError` in plain Python scripts) that renders the small 2-row table nicely.
- Bar chart comparing accuracy and macro-F1 side by side for zero-shot vs fine-tuned.
- **Expected output:** a rendered 2×3 table, then a bar chart with 2 model groups × 2 metrics each, both metrics for fine-tuned should clearly exceed both metrics for zero-shot, and both should exceed the 0.5 random-baseline line.

---

## Section 5 — Experiment 3: The Chained Pipeline

### Cell: Build the chain test set
```python
chain_test = test_df[test_df[EDITED_COL].astype(str).str.len() > 10].copy()
print(f"Chain test examples (with valid sentence2): {len(chain_test)}")
chain_test[[PASSAGE_COL, EDITED_COL, QUESTION_COL, "answer_clean"]].head(3)
```
- Filters the (sampled, 200-row) `test_df` down to rows where `sentence2` (the edited/negation-flipped counterfactual passage) is non-trivial (more than 10 characters) — guards against empty or placeholder values.
- **Expected output:** a count (likely close to 200, possibly slightly less if some rows have missing/short `sentence2`), and a 3-row preview table showing original passage, edited passage, question, and answer.

### Cell: Run the chained pipeline
```python
model.eval()

def model_predict(question, passage):
    enc = tokenizer(question, passage, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return int(torch.argmax(logits, dim=1).item())

chain_results = []
for _, row in chain_test.iterrows():
    q  = str(row[QUESTION_COL])
    d1 = str(row[PASSAGE_COL])
    d2 = str(row[EDITED_COL])
    gt = row["label_int"]

    d1_tok = clean_bm25(d1); d2_tok = clean_bm25(d2)
    ret_idx = bm25_pick(clean_bm25(q), [d1_tok, d2_tok])
    retrieved_doc = d1 if ret_idx == 0 else d2
    retrieval_correct = (ret_idx == 0)

    pred = model_predict(q, retrieved_doc)
    pipeline_correct = (pred == gt)

    chain_results.append({
        "retrieval_correct":  retrieval_correct,
        "model_pred":         pred,
        "ground_truth":       gt,
        "pipeline_correct":   pipeline_correct,
    })

chain_df = pd.DataFrame(chain_results)
chain_retrieval_acc = chain_df["retrieval_correct"].mean()
chain_pipeline_acc  = chain_df["pipeline_correct"].mean()
predicted_acc       = chain_retrieval_acc * ft_acc

print(f"Retrieval accuracy (this subset)  : {chain_retrieval_acc:.3f}")
print(f"Model accuracy (isolated, ft)     : {ft_acc:.3f}")
print(f"PREDICTED pipeline accuracy       : {predicted_acc:.3f}")
print(f"ACTUAL pipeline accuracy          : {chain_pipeline_acc:.3f}")
print(f"Gap (predicted - actual)          : {predicted_acc - chain_pipeline_acc:+.3f}")
```
- This is **the key experiment**. For every CondaQA test example, it builds a mini 2-document corpus: the true `original passage` (correct evidence) and `sentence2` (the negation-flipped edited passage, used here as a "distractor").
- Runs the same BM25 `clean_bm25`/`bm25_pick` pipeline from Experiment 1 to pick which of the two documents to retrieve for this question. `retrieval_correct` is `True` only if BM25 picked the true original passage (index 0).
- Whichever document BM25 picked (`retrieved_doc`) — right or wrong — gets handed to the **fine-tuned** DistilBERT model (`model_predict`) along with the question, to produce a yes/no prediction.
- `pipeline_correct` = whether that final prediction matches the true ground-truth answer, **regardless of whether the model was shown the right document**.
- After the loop:
  - `chain_retrieval_acc` = fraction of examples where BM25 retrieved the correct passage.
  - `chain_pipeline_acc` = fraction of examples where the **end-to-end pipeline** got the final answer right.
  - `predicted_acc` = the naive "if errors were independent" estimate: retrieval accuracy × model's isolated accuracy (`ft_acc` from Experiment 2). This is the theoretical benchmark: if the model somehow always answered correctly whenever the model itself would be accurate on that passage, and retrieval errors were unrelated to whether the model would get it right, you'd expect this product.
  - The **gap** (`predicted_acc - chain_pipeline_acc`) is the notebook's headline metric: if positive, the pipeline performs worse than a naive independence assumption predicts — i.e., the errors compound rather than being independent.
- **Expected output:** five printed lines with retrieval accuracy (likely similar to Experiment 1's near-random BM25 result, since it's the same flawed cleaning pipeline), the fine-tuned model's isolated accuracy (reused from Experiment 2), the predicted product of the two, the actual measured pipeline accuracy, and the gap. The notebook's hypothesis predicts the gap will be **positive** (actual noticeably lower than predicted), supporting the "compounding errors" thesis — but the real numeric outcome depends on this specific run's data and model.

### Cell: Predicted vs Actual chart + error breakdown
```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bars = axes[0].bar(["Predicted\n(independence)", "Actual\n(chained pipeline)"],
                    [predicted_acc, chain_pipeline_acc], ...)
for b in bars:
    axes[0].text(..., f"{b.get_height():.3f}", ...)
...
chain_df["ret_ok"]   = chain_df["retrieval_correct"]
chain_df["model_ok"] = (chain_df["model_pred"] == chain_df["ground_truth"])
breakdown = {
    "Ret ✓  Mdl ✓": ...,
    "Ret ✓  Mdl ✗": ...,
    "Ret ✗  Mdl ✓": ...,
    "Ret ✗  Mdl ✗": ...,
}
axes[1].bar(breakdown.keys(), breakdown.values(), ...)
plt.savefig("experiment3_main.png", dpi=120)
plt.show()
```
- Left: a simple 2-bar chart of predicted vs actual pipeline accuracy, with the exact value labeled above each bar.
- Right: builds a 2×2 breakdown of outcomes — retrieval correct/incorrect crossed with the final model prediction being correct/incorrect vs ground truth — as a 4-bar chart with counts labeled above each bar.
- **Expected output:** the headline bar chart (this is the notebook's main deliverable figure) plus the 4-category breakdown. The "Ret ✗ Mdl ✓" bar (wrong document retrieved but the model still lucked into the right answer) being small, and "Ret ✗ Mdl ✗" being large, would support the compounding-errors story.

### Cell: Confusion matrices — isolated model vs full pipeline
```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
cm_isolated = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm_isolated, ..., ax=axes[0])
cm_chain = confusion_matrix(chain_df["ground_truth"], chain_df["model_pred"])
sns.heatmap(cm_chain, ..., cmap="Oranges", ax=axes[1])
plt.savefig("experiment3_confusion.png", dpi=120)
plt.show()
print("Comparing these two matrices shows exactly how retrieval errors")
print("change the model's error pattern — not just overall accuracy.")
```
- Left: reuses the fine-tuned model's confusion matrix from Experiment 2 (model always given the correct passage).
- Right: confusion matrix for the full chained pipeline (model given whatever BM25 retrieved, right or wrong), using a different color map (orange) to visually distinguish it.
- **Expected output:** two heatmaps side by side. The right (orange) one should show a noticeably weaker diagonal (more off-diagonal errors) than the left (blue) one, visually demonstrating the accuracy drop caused by retrieval errors feeding into the model.

### Cell: Full experiment summary chart
```python
summary = pd.DataFrame({
    "Stage": [
        "BM25 retrieval\n(single query)",
        "Dense MiniLM\n(single query)",
        "DistilBERT\nzero-shot",
        "DistilBERT\nfine-tuned",
        "Pipeline PREDICTED\n(independence)",
        "Pipeline ACTUAL\n(chained)",
    ],
    "Accuracy": [bm25_single, dense_single, zs_acc, ft_acc, predicted_acc, chain_pipeline_acc],
    "Stage type": ["Retrieval","Retrieval","Generation","Generation","Pipeline","Pipeline"]
})

color_map = {"Retrieval":"#378ADD","Generation":"#0F6E56","Pipeline":"#D85A30"}
colors = [color_map[s] for s in summary["Stage type"]]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(summary["Stage"], summary["Accuracy"], color=colors, edgecolor="white")
ax.axhline(0.5, color="gray", linestyle="--", lw=1.2, label="Random baseline (0.5)")
for b in bars:
    ax.text(..., f"{b.get_height():.2f}", ...)
plt.savefig("full_summary.png", dpi=120)
plt.show()
```
- Assembles every accuracy number computed across the whole notebook (retrieval stage, generation stage, and pipeline stage) into a single 6-bar comparison chart, color-coded by stage type (blue=retrieval, green=generation, orange=pipeline), each bar labeled with its value, with a gray dashed random-baseline reference line.
- **Expected output:** a single wide bar chart that functions as the notebook's "big picture" figure — comparing all 6 accuracy numbers computed throughout the notebook side by side.

---

## Section 6 — Conclusions (markdown only, no code)

This final section is prose, not code: it summarizes the expected findings (BM25 near-random on negation pairs, dense embeddings also struggling but less severely, fine-tuning helping the generation stage, and the chained pipeline underperforming the independence-assumption prediction), lists honest caveats (small `TRAIN_SIZE`, DistilBERT not being a frontier model, `sentence2` being an imperfect distractor), and suggests follow-up work (larger data, comparing dense retrieval in the chain, trying a frontier LLM as generator).

---

## Practical notes for running this yourself

- **Runtime:** on a free Colab T4 GPU, expect maybe 5–15 minutes total (dataset downloads + BM25/dense retrieval loops + 4 epochs of DistilBERT fine-tuning + the chain loop). On CPU-only, fine-tuning will be the slow part — could take significantly longer.
- **Non-determinism:** although seeds are set, exact numbers can still vary slightly run-to-run due to library/hardware nondeterminism (e.g., cuDNN, MiniLM download version).
- **`display()`** cell will error outside Jupyter/Colab (e.g., if converted to a plain `.py` script) since it's an IPython builtin, not standard Python — replace with `print(improve_df)` if running as a script.
- **Column names** for NevIR/CondaQA are dataset-version-dependent — if you get a `KeyError` from `pick()` or on `PASSAGE_COL`/`QUESTION_COL`, re-run the cells right before them that print `list(df.columns)` to see the actual names and adjust the constants.
