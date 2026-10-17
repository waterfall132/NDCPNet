#!/usr/bin/env python3
"""CPU tests using project model code; no checkpoint or dataset required."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import d2_joint_head as h


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    from models.ndcn import NDCN_Metric
    for shot in (1, 5):
        fq = torch.rand(7, 12, 5, 5, dtype=torch.double, requires_grad=True)
        fs = torch.rand(5*shot, 12, 5, 5, dtype=torch.double, requires_grad=True)
        out = h.d2_scores(fq, fs, shot=shot)
        p = fs.mean((2,3)).reshape(5,shot,-1).mean(1)
        z = fq.flatten(2).transpose(1,2)
        sim = F.normalize(z,dim=-1) @ F.normalize(p,dim=-1).T
        direct = torch.stack([sim[:,:,c] - sim[:,:,[k for k in range(5) if k!=c]].mean(-1) for c in range(5)], -1)
        torch.testing.assert_close(out['contrast'],direct)
        torch.testing.assert_close(out['weights'].sum(1),torch.ones(7,5,dtype=torch.double))
        check(torch.equal(h.d2_scores(fq,fs,shot=shot,weight=0)['d2'],h.gap_scores(fq,fs,shot=shot)), 'lambda=0 must exactly recover GAP')
        classes = torch.tensor([3,0,4,1,2])
        perm_s = fs.reshape(5,shot,12,5,5)[classes].reshape_as(fs)
        torch.testing.assert_close(h.d2_scores(fq,perm_s,shot=shot)['d2'],out['d2'][:,classes])
        torch.testing.assert_close(h.d2_scores(fq[:1],fs,shot=shot)['d2'],out['d2'][:1])
        if shot == 5:
            swapped=fs.reshape(5,shot,12,5,5).flip(1).reshape_as(fs)
            torch.testing.assert_close(h.d2_scores(fq,swapped,shot=shot)['d2'],out['d2'])
        loss=F.cross_entropy(10*out['d2'], torch.arange(7)%5)
        gradients=torch.autograd.grad(loss,(fq,fs,out['weights']))
        check(all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients),'Missing finite gradients to query/support/attention')
        check(1 <= float(out['effective_positions'].detach()) <= 25.00001,'Invalid effective positions')
        # Old GAP inference/training score equivalence (float tolerance for reductions).
        old=NDCN_Metric(way_num=5,shot_num=shot,uniform_mask=True,use_bg_sep=False,use_qbg=False,
                       use_triplet=False,use_bgloss=False,use_div=False,use_mask=False).double()
        old.ndm.forward=lambda *a,**k: (_ for _ in ()).throw(AssertionError('NDM called'))
        y=torch.arange(7)%5
        old.train()
        legacy,_=old(fq,fs,y)
        expected=old.gamma_fg*(h.gap_scores(fq,fs,shot=shot)-1)
        torch.testing.assert_close(legacy,expected,atol=1e-10,rtol=1e-10)
        print('PASS: shot=%d formula, spatial softmax, class permutation, query independence at head, GAP equivalence and gradients' % shot)
    from models import network
    import Train_R12_D2_joint as trainer
    cfg=json.loads(Path(__file__).with_name('r12_d2_joint_settings.json').read_text())
    cfg.update(trainer.arm_settings('GAP'))
    opt=argparse.Namespace(**cfg)
    model=network.define_model(encoder_model='ResNet12_ReLU',classifier_model='NDCN',
                              way_num=5,shot_num=1,use_gpu=False,ndcn_cfg=trainer.build_ndcn_cfg(opt))
    # Real backbone: compare GAP legacy full forward against new feature-wrapper path,
    # including parameter gradients; then check actual D2 backbone update.
    model.classifier.ndm.forward=lambda *a,**k: (_ for _ in ()).throw(AssertionError('NDM called'))
    q,s=torch.randn(2,3,84,84),torch.randn(5,3,84,84)
    y=torch.tensor([0,1])
    clone=copy.deepcopy(model)
    torch.manual_seed(19)
    legacy,_=model(q,s,y)
    F.cross_entropy(legacy,y).backward()
    torch.manual_seed(19)
    fq,fs=clone.features(q),clone.features(s)
    new=h.training_logits(clone,fq,fs,y,'GAP',1)
    F.cross_entropy(new,y).backward()
    torch.testing.assert_close(new,legacy,atol=0,rtol=0)
    for (n,p),(m,r) in zip(model.named_parameters(),clone.named_parameters()):
        check(n==m,'Parameter order mismatch')
        if p.grad is None:
            check(r.grad is None,'Unexpected parameter gradient')
        else:
            torch.testing.assert_close(p.grad,r.grad,atol=0,rtol=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5,betas=(.5,.9),weight_decay=5e-4)
    optimizer.zero_grad(set_to_none=True)
    before=next(model.features.parameters()).detach().clone()
    logits=h.training_logits(model,model.features(q),model.features(s),y,'D2',1)
    F.cross_entropy(logits,y).backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
    check(torch.isfinite(norm),'Nonfinite backbone gradient')
    optimizer.step()
    check(not torch.equal(before,next(model.features.parameters())),'Backbone did not update')
    check(model.classifier._raw_gamma_fg.grad is not None,'Logit scale did not get gradient')
    print('PASS: real ResNet-12 GAP logits/gradients exactly match old B; D2 updates backbone; no NDM calls')
    # Optionally verify the existing frozen D2 implementation if supplied.
    parser=argparse.ArgumentParser()
    parser.add_argument('--frozen-head',default='')
    args=parser.parse_args()
    if args.frozen_head:
        spec=importlib.util.spec_from_file_location('frozen',args.frozen_head)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fq,fs=torch.rand(7,12,5,5),torch.rand(5,12,5,5)
        original=module.GAPLocalEvidence(None,temperature=.2)(fq,fs,branch='D2',weight=.25)
        torch.testing.assert_close(original,h.d2_scores(fq,fs)['d2'],atol=0,rtol=0)
        print('PASS: D2 forward exactly matches the previous frozen-screen implementation')
    print('ALL TESTS PASSED (CPU correctness tests, not miniImageNet performance evidence).')


if __name__=='__main__':
    main()
