"""Real multi-category text for experiment 3: AG News
(https://huggingface.co/datasets/fancyzhx/ag_news), 120k real news snippets in
4 categories (world / sports / business / sci-tech). Each category is a
"tenant" with its own real vocabulary: no synthetic text anywhere here.

Rather than construct a "tenant-common but globally rare" word by hand, we
*discover* real examples of it: for each category, find words that show up
often in that category's own documents but are almost never used by the other
three (the "domain-common" candidates, e.g. "minister" for World news), and
words that are rare even within their own category (the "domain-rare"
candidates, e.g. "hostage", the kind of specific term a real query is
actually looking for). Both are measured, not chosen for a story.
"""

import re
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "fancyzhx/ag_news"
FILENAME = "data/train-00000-of-00001.parquet"
CACHE_DIR = "data/.hf_cache"
CATEGORIES = ["world", "sports", "business", "scitech"]  # AG News label order 0-3

TOKEN_RE = re.compile(r"[a-z]{3,}")

# Generic-enough-to-be-uninteresting-as-an-example words: statistically valid
# candidates, but not what a reader pictures as "domain jargon". Excluded only
# from which words get *picked* to headline the demo; the underlying
# frequency stats are computed over the real, un-filtered vocabulary.
GENERIC_STOPLIST = {
    "www", "href", "com", "the", "and", "for", "that", "with", "this", "from",
    "has", "have", "will", "are", "was", "were", "its", "his", "her", "their",
    "target", "available", "size", "system", "web",
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_ag_news_sample(n_per_category, seed):
    """Returns a DataFrame with columns: category, tokens (list[str])."""
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, cache_dir=CACHE_DIR)
    df = pd.read_parquet(path)

    frames = []
    for label, category in enumerate(CATEGORIES):
        sub = df[df.label == label].sample(n=n_per_category, random_state=seed + label)
        sub = sub.assign(category=category)
        frames.append(sub)
    sample = pd.concat(frames, ignore_index=True)[["category", "text"]]
    sample["tokens"] = sample["text"].apply(tokenize)
    return sample


def _doc_freq_by_category(sample, categories):
    doc_freq = {c: Counter() for c in categories}
    doc_count = {c: 0 for c in categories}
    for category, tokens in zip(sample["category"], sample["tokens"]):
        if category not in doc_freq:
            continue
        doc_count[category] += 1
        for tok in set(tokens):
            doc_freq[category][tok] += 1
    return doc_freq, doc_count


def discover_word_pairs(
    sample,
    categories,
    top_k=6,
    common_frac_range=(0.04, 0.30),
    rare_frac_range=(0.004, 0.035),
    max_other_frac=0.01,
):
    """For each category, returns {"domain_common": [...], "domain_rare": [...]}:
    real words meeting the frequency criteria in the docstring above,
    ranked by within-category frequency, generic filler excluded."""
    doc_freq, doc_count = _doc_freq_by_category(sample, categories)
    result = {}
    for category in categories:
        others = [c for c in categories if c != category]
        common, rare = [], []
        for tok, count in doc_freq[category].items():
            if tok in GENERIC_STOPLIST:
                continue
            frac_here = count / doc_count[category]
            frac_others = max((doc_freq[o].get(tok, 0) / doc_count[o]) for o in others)
            if frac_others > max_other_frac:
                continue
            if common_frac_range[0] <= frac_here <= common_frac_range[1]:
                common.append((tok, frac_here))
            if rare_frac_range[0] <= frac_here <= rare_frac_range[1]:
                rare.append((tok, frac_here))
        common.sort(key=lambda x: -x[1])
        rare.sort(key=lambda x: -x[1])
        result[category] = {
            "domain_common": [w for w, _ in common[:top_k]],
            "domain_rare": [w for w, _ in rare[:top_k]],
        }
    return result


def tokens_to_sparse(tokens, word_to_id):
    counts = {}
    for tok in tokens:
        if tok in word_to_id:
            wid = word_to_id[tok]
            counts[wid] = counts.get(wid, 0) + 1
    return counts
