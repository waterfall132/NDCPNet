"""D2 joint training: fixed class-contrast attention, no NDM or auxiliary loss.

Inputs are feature maps, never query labels. All support/query operations remain
in the autograd graph. T is a spatial softmax temperature, not logit scaling.
"""
import torch
import torch.nn.functional as F

WEIGHT = 0.25
TEMPERATURE = 0.2


def gap_scores(fq, fs, way=5, shot=1):
    p = fs.mean((2, 3)).reshape(way, shot, -1).mean(1)
    return F.normalize(fq.mean((2, 3)), dim=-1) @ F.normalize(p, dim=-1).T


def d2_scores(fq, fs, way=5, shot=1, weight=WEIGHT, temperature=TEMPERATURE):
    if fq.ndim != 4 or fs.ndim != 4 or fq.shape[1] != fs.shape[1]:
        raise ValueError('Expected query/support feature maps of shape B,C,H,W')
    if way < 2 or shot < 1 or fs.shape[0] != way * shot:
        raise ValueError('Support maps must be class-major, way*shot')
    if not 0 <= weight <= 1 or temperature <= 0:
        raise ValueError('Invalid mixture weight or spatial temperature')
    gap = gap_scores(fq, fs, way, shot)
    if weight == 0:
        return {'gap': gap, 'd2': gap}
    # Mean over shots BEFORE normalization: equal weight for each support image.
    prototypes = fs.mean((2, 3)).reshape(way, shot, -1).mean(1)
    z = fq.flatten(2).transpose(1, 2).contiguous()  # Q,L,C; pool original (not normalized) z.
    p_unit = F.normalize(prototypes, dim=-1)
    sim = F.normalize(z, dim=-1) @ p_unit.T  # Q,L,N
    contrast = sim - (sim.sum(-1, keepdim=True) - sim) / (way - 1)
    weights = (contrast / temperature).softmax(dim=1)  # over SPACE, not classes
    local_vectors = weights.transpose(1, 2) @ z  # Q,N,C
    local = (F.normalize(local_vectors, dim=-1) * p_unit.unsqueeze(0)).sum(-1)
    mixed = (1 - weight) * gap + weight * local
    return {'gap': gap, 'd2': mixed, 'local': local, 'weights': weights,
            'contrast': contrast, 'local_vectors': local_vectors,
            'effective_positions': weights.square().sum(1).reciprocal().mean()}


def training_logits(model, fq, fs, y, arm, shot):
    """GAP retains legacy B's exact head implementation; D2 uses its same scale."""
    if arm == 'GAP':
        logits, aux = model.classifier(fq, fs, y)
        if any(not torch.isfinite(v).all() or abs(float(v.detach())) > 1e-12
               for v in aux.values()):
            raise RuntimeError('GAP baseline must have zero auxiliary losses')
        return logits
    if arm != 'D2':
        raise ValueError('Unknown arm')
    mixed = d2_scores(fq, fs, way=5, shot=shot)['d2']
    # Identical convention to the old head: -gamma * cosine_distance.
    # The class-common offset -gamma does not change CE or predictions.
    return model.classifier.gamma_fg * (mixed - 1.0)
