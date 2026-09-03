"""Synthetic multi-tenant text corpora for the per-tenant IDF experiment.

Each tenant is a different "industry" with its own vocabulary:

  - COMMON words: shared across every tenant at similar frequency (the
    generic "please help with my account" filler every support ticket has).
  - DOMAIN-COMMON words: exclusive to one tenant, and frequent *within* that
    tenant (e.g. "invoice" shows up in half of a fintech tenant's tickets).
    Nothing else in the collection ever uses these words.
  - DOMAIN-RARE words: exclusive to one tenant, and genuinely rare even
    there: the actually-informative terms a real query is looking for.

This is what makes the distortion in global IDF concrete and checkable: a
domain-common word's raw document frequency is identical whether you count
it globally or within the tenant (nobody else ever uses it), but the
*denominator* (total corpus size) is much bigger globally than within one
tenant. log(N/df) with the big global N makes a word that is actually
common-and-boring within its own tenant look rare-and-important, which
misdirects BM25-style ranking. Per-tenant IDF fixes the denominator (and
the count) to the tenant's own corpus.
"""

import numpy as np


def build_vocab_and_tenants(tenants, n_common, n_domain_common, n_domain_rare, seed):
    rng = np.random.default_rng(seed)
    vocab = []
    common_words = [f"common_{i}" for i in range(n_common)]
    vocab += common_words

    tenant_vocab = {}
    for t in tenants:
        domain_common = [f"{t}_common_{i}" for i in range(n_domain_common)]
        domain_rare = [f"{t}_rare_{i}" for i in range(n_domain_rare)]
        vocab += domain_common + domain_rare
        tenant_vocab[t] = {"domain_common": domain_common, "domain_rare": domain_rare}

    word_to_id = {w: i for i, w in enumerate(vocab)}
    return word_to_id, common_words, tenant_vocab


def generate_tenant_docs(tenant, common_words, domain_common, domain_rare, n_docs, doc_len_range, domain_common_prob, domain_rare_prob, seed):
    """Returns a list of dicts: {"tokens": [...], "rare_hits": set(word)}.

    domain_common[i] and domain_rare[i] are paired: a ticket that actually
    raises the rare/specific issue doesn't also spam the generic jargon term
    for that same topic, so the two "explain" the same underlying issue at
    different specificity rather than stacking. This keeps the two signals
    separable instead of the same handful of tickets containing both.
    """
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_docs):
        length = rng.integers(doc_len_range[0], doc_len_range[1] + 1)
        tokens = list(rng.choice(common_words, size=length, replace=True))
        rare_hits = set()
        for w_common, w_rare in zip(domain_common, domain_rare):
            if rng.random() < domain_rare_prob:
                tokens.append(w_rare)
                rare_hits.add(w_rare)
            elif rng.random() < domain_common_prob:
                # domain-common jargon tends to get repeated within a ticket
                # ("invoice ... invoice number ... invoice date ..."), unlike
                # the one-off rare term. That repetition is what makes an
                # inflated global IDF weight actually bite in the ranking.
                tokens += [w_common] * int(rng.integers(1, 4))
        rng.shuffle(tokens)
        docs.append({"tenant": tenant, "tokens": tokens, "rare_hits": rare_hits})
    return docs


def tokens_to_sparse(tokens, word_to_id):
    """Raw term-frequency sparse vector: {word_id: count}."""
    counts = {}
    for tok in tokens:
        if tok in word_to_id:
            wid = word_to_id[tok]
            counts[wid] = counts.get(wid, 0) + 1
    return counts
