import numpy as np

def mask_and_classify_scores(items):
    if not isinstance(items, np.ndarray):
        return None
    if items.ndim != 2:
        return None
    total, bound = items.shape
    if total != bound or total < 4:
        return None

    cleaned = items.copy()
    cleaned[cleaned < 0] = 0

    cleaned[cleaned > 100] = 100

    levels = np.zeros_like(cleaned, dtype=int)
    levels[(cleaned >= 40) & (cleaned < 70)] = 1
    levels[cleaned >= 70] = 2

    row_pass_counts = np.zeros(total, dtype=int)
    for idx in range(total):
        count = 0
        for col in range(total):
            if cleaned[idx, col] >= 50:
                count += 1
        row_pass_counts[idx] = count

    return (cleaned, levels, row_pass_counts)