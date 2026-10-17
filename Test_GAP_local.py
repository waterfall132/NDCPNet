#!/usr/bin/env python3
"""CPU mathematical/invariance tests using the project's real NDM. No data needed."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main(root):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ndp_gap_local import (GAPLocalEvidence, BRANCHES, gap_scores,
                              discrimination_scores, smooth_evidence, attention_scores, mix_scores)
    from evidence_stats import summarize, choose
    torch.set_num_threads(2)
    torch.manual_seed(24)
    spec=importlib.util.spec_from_file_location('test_original_ndcn',Path(root)/'models/ndcn.py')
    original=importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    head=GAPLocalEvidence(original.NDM(rank=3,n_iter=3)).eval()
    q=torch.rand(7,12,3,3); s=torch.rand(5,12,3,3)
    scores,_=head.branches(q,s)
    assert torch.equal(head(q,s,'D3',0),gap_scores(q,s))
    assert torch.equal(mix_scores(scores['D0'],scores['D2'],0),scores['D0'])
    print('PASS exact lambda=0 GAP recovery')
    # All branch predictions are independent of the other queries in the batch.
    singles=[head.branches(q[i:i+1],s)[0] for i in range(len(q))]
    for name in scores:
        torch.testing.assert_close(scores[name],torch.cat([x[name] for x in singles]),atol=2e-5,rtol=2e-5)
    print('PASS query-batch independence (real NDM)')
    perm=torch.tensor([2,4,0,3,1])
    reordered,_=head.branches(q,s[perm])
    for name in scores:
        torch.testing.assert_close(reordered[name],scores[name][:,perm],atol=2e-5,rtol=2e-5)
    print('PASS support-class permutation equivariance')
    # Exact legacy D1 branch, without its positive temperature or common offset.
    old=original.NDCN_Metric(way_num=5,shot_num=1,rank=3,n_iter=3,
        score_mode='free',alpha_norm='supminmax',use_bg_sep=False,use_qbg=False)
    old.eval()
    legacy=old(q,s)/old.gamma_fg+1
    torch.testing.assert_close(legacy,scores['D1'],atol=2e-5,rtol=2e-5)
    print('PASS D1 equals original masked classifier cosine scores')
    # Independent NumPy implementation of discrimination -> smoothing -> attention.
    z=q.flatten(2).transpose(1,2); p=s.mean((2,3)); a,_,_=head.ndm(z)
    zn=z.detach().numpy(); pn=p.numpy(); an=a.detach().numpy()
    normalize=lambda x:x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    sim=normalize(zn)@normalize(pn).T
    dn=sim-(sim.sum(-1,keepdims=True)-sim)/4
    smooth=an@(np.swapaxes(an,1,2)@dn/(an.sum(1)[...,None]+1e-8))
    for name,ev in [('D2',dn),('D3',smooth)]:
        logits=ev/.2; logits=logits-logits.max(1,keepdims=True)
        w=np.exp(logits); w=w/w.sum(1,keepdims=True)
        vectors=np.swapaxes(w,1,2)@zn
        ref=(normalize(vectors)*normalize(pn)[None]).sum(-1)
        np.testing.assert_allclose(scores[name].detach().numpy(),ref,atol=2e-6,rtol=2e-6)
    print('PASS independent NumPy D2/D3 reference')
    # Equal component memberships should yield constant evidence and GAP.
    uniform=torch.full((7,9,3),1/3)
    evidence=smooth_evidence(discrimination_scores(z,p),uniform)
    result,w=attention_scores(z,p,evidence)
    torch.testing.assert_close(result,scores['D0'],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(w,torch.full_like(w,1/9),atol=1e-6,rtol=1e-6)
    zeros,_=head.branches(torch.zeros_like(q),torch.zeros_like(s))
    assert all(torch.isfinite(x).all() for x in zeros.values())
    print('PASS constant evidence -> GAP; all-zero features remain finite')
    # Head can be used by a future CE trainer; no silent feature detachment.
    for branch in ('D2','D3'):
        gq=q.clone().requires_grad_(True); gs=s.clone().requires_grad_(True)
        loss=F.cross_entropy(10*head(gq,gs,branch,.25),torch.arange(7)%5)
        loss.backward()
        assert torch.isfinite(gq.grad).all() and gq.grad.abs().sum()>0
        assert torch.isfinite(gs.grad).all() and gs.grad.abs().sum()>0
    print('PASS finite nonzero query/support gradients for CE')
    targets=np.zeros((3,4),dtype=int)
    preds={'D0':np.zeros_like(targets),'D2_w0.1':np.zeros_like(targets)}
    stats,_=summarize(preds,targets); assert choose(stats)=='D0'
    preds['D2_w0.1'][0,0]=1
    stats,_=summarize(preds,targets)
    assert stats['results']['D2_w0.1']['broken_query_occurrences']==1
    print('PASS task statistics and baseline tie priority')
    print('[PASS] All tests complete. Real checkpoint/data smoke remains required.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default='.')
    main(p.parse_args().project_root)
