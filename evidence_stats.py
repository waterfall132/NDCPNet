"""Task-level paired statistics. No pooling across training seeds."""
import numpy as np
from scipy.stats import t


def interval(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need >=2 finite task scores')
    h = t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), ci95_low=float(x.mean()-h),
                ci95_high=float(x.mean()+h))


def summarize(predictions, targets):
    targets = np.asarray(targets)
    if targets.ndim != 2:
        raise ValueError('Targets must be tasks x query occurrences')
    if 'D0' not in predictions:
        raise ValueError('Missing GAP baseline')
    if any(np.asarray(v).shape != targets.shape for v in predictions.values()):
        raise ValueError('Unpaired predictions')
    correct = {k: np.asarray(v) == targets for k, v in predictions.items()}
    scores = {k: v.mean(1)*100 for k, v in correct.items()}
    results = {}
    for k, x in scores.items():
        repaired = ~correct['D0'] & correct[k]
        broken = correct['D0'] & ~correct[k]
        results[k] = dict(accuracy=interval(x), minus_D0=interval(x-scores['D0']),
                          corrected_query_occurrences=int(repaired.sum()),
                          broken_query_occurrences=int(broken.sum()))
    contrasts = {}
    for w in (0.1, 0.25):
        for a, b in [('D3','D2'), ('D2','D1q'), ('D1q','D1')]:
            ka, kb = f'{a}_w{w:g}', f'{b}_w{w:g}'
            if ka in scores and kb in scores:
                contrasts[f'{ka}_minus_{kb}'] = interval(scores[ka]-scores[kb])
    return dict(results=results, controlled_contrasts=contrasts), scores


def choose(summary):
    # Insertion order gives D0 priority in an exact tie; then smaller weights.
    return max(summary['results'], key=lambda k: summary['results'][k]['accuracy']['mean'])
