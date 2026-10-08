"""Retrieval metrics: recall@k, average precision, mAP."""


def recall_at_k(ranking, relevant_ids, k):
    """ranking: list of ids ordered by score desc; relevant_ids: set of ids."""
    if not relevant_ids:
        return 0.0
    top = set(ranking[:k])
    return len(top & set(relevant_ids)) / len(relevant_ids)


def average_precision(ranking, relevant_ids):
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    for i, rid in enumerate(ranking, 1):
        if rid in relevant_ids:
            hits += 1
            score += hits / i
    return score / len(relevant_ids)


def mean_average_precision(rankings, relevant_map):
    """rankings: list of ranked id lists; relevant_map: {qid: set of relevant ids}."""
    if not rankings:
        return 0.0
    aps = [average_precision(rankings[i], relevant_map.get(i, set())) for i in range(len(rankings))]
    return sum(aps) / len(aps)
