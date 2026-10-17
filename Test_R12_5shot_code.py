#!/usr/bin/env python3
"""No-data CPU tests for the changed five-shot tensor and prototype paths."""
import argparse
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
import Train_NDCN_5shot_check as trainer


def main(root):
    trainer.torch=torch; trainer.F=F
    torch.set_num_threads(2);torch.manual_seed(6)
    spec=importlib.util.spec_from_file_location('ndcn_original',Path(root)/'models/ndcn.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    q=torch.rand(75,12,3,3);s=torch.rand(25,12,3,3)
    p=torch.stack([s[c*5:(c+1)*5].mean((0,2,3)) for c in range(5)])
    ref=F.normalize(q.mean((2,3)),dim=1)@F.normalize(p,dim=1).T
    got=trainer.gap_logits(q,s)
    assert got.shape==(75,5)
    torch.testing.assert_close(got,ref,atol=2e-6,rtol=2e-6)
    old=mod.NDCN_Metric(way_num=5,shot_num=5,rank=3,n_iter=3,uniform_mask=True,
        score_mode='free',alpha_norm='supminmax',use_bg_sep=False,use_qbg=False,
        use_triplet=False,use_div=False,use_mask=False,use_bgloss=False)
    old.eval(); legacy=old(q,s)/old.gamma_fg+1
    torch.testing.assert_close(got,legacy,atol=2e-6,rtol=2e-6)
    print('PASS five-shot GAP: five class prototypes, matches independent mean and uniform classifier')
    # Permute support images WITHIN each class: prototypes must stay unchanged.
    reordered=s.reshape(5,5,12,3,3).flip(1).reshape_as(s)
    torch.testing.assert_close(got,trainer.gap_logits(q,reordered),atol=2e-6,rtol=2e-6)
    print('PASS within-class support permutation invariance')
    qi=[torch.zeros(1,3,84,84) for _ in range(75)]
    qt=[torch.tensor([i//15]) for i in range(75)]
    si=[torch.zeros(1,5,3,84,84) for _ in range(5)]
    st=[torch.tensor([i//5]) for i in range(25)]
    b=trainer.batch_tensors((qi,qt,si,st),'cpu')
    assert b[0].shape==(75,3,84,84) and b[1].shape==(25,3,84,84)
    try: trainer.batch_tensors((qi,qt,si,list(reversed(st))),'cpu')
    except ValueError: pass
    else: raise AssertionError('Incorrect support class order accepted')
    print('PASS 25-support collation and class-label guard')
    old.uniform_mask=False;old.use_triplet=True;old.use_div=True;old.use_mask=True;old.train()
    q.requires_grad_();s.requires_grad_()
    logits,aux=old(q,s,torch.arange(75)//15)
    loss=F.cross_entropy(logits,torch.arange(75)//15)+.5*aux['total']
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()
    print('PASS original five-shot mask head: finite forward/loss/backward')
    print('[PASS] Five-shot code checks complete; real data/GPU training not executed by this test.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-root',default='.')
    main(p.parse_args().project_root)
