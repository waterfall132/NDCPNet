#!/usr/bin/env python3
"""Check actual SGD updates and original NDPNet/GAP gradients; no dataset needed."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import Train_R12_SGD_AB as tr


def main():
    torch.set_num_threads(2)
    tr.torch = torch
    a = argparse.Namespace(optimizer='sgd',lr=.001,momentum=.9,dampening=.9)
    p = torch.nn.Parameter(torch.tensor([1.0,2.0],dtype=torch.float64))
    opt = tr.make_optimizer([{'params':[p],'weight_decay':.0005}],a)
    assert isinstance(opt,torch.optim.SGD) and not isinstance(opt,torch.optim.Adam)
    expected=p.detach().clone(); buf=None
    for i in range(3):
        grad=torch.tensor([.2+i,-.1-i],dtype=torch.float64)
        p.grad=grad.clone()
        g=grad+.0005*expected
        buf=g.clone() if buf is None else .9*buf+.1*g
        expected=expected-.001*buf
        opt.step()
        torch.testing.assert_close(p,expected,rtol=1e-12,atol=1e-12)
    assert 'momentum_buffer' in opt.state[p] and 'exp_avg' not in opt.state[p]
    print('PASS: real SGD, original momentum/dampening update formula, no Adam state')
    from models.ndcn import NDCN_Metric
    for shot in (1,5):
        for arm in ('A','B'):
            torch.manual_seed(5)
            flags=tr.arm_settings(arm)
            head=NDCN_Metric(way_num=5,shot_num=shot,uniform_mask=flags['uniform_mask'],
                use_triplet=not flags['no_triplet'],use_div=not flags['no_div'],use_mask=not flags['no_mask'],
                use_bg_sep=False,use_qbg=False,use_bgloss=False,score_mode='free',alpha_norm='supminmax',
                rank=5,n_iter=3,mask_budget=.25,lambda_div=.2,lambda_mask=.01)
            head.train()
            q=torch.rand(7,12,5,5,requires_grad=True)
            s=torch.rand(5*shot,12,5,5,requires_grad=True)
            y=torch.arange(7)%5
            logits,aux=head(q,s,y)
            loss=F.cross_entropy(logits,y)
            if arm=='A':loss=loss+.5*aux['total']
            else:assert all(float(v.detach())==0 for v in aux.values())
            loss.backward()
            assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
            assert torch.isfinite(s.grad).all() and s.grad.abs().sum()>0
            print('PASS: shot=%d arm=%s original head and auxiliary-loss flags, finite support/query gradients' % (shot,arm))
    from models import network
    cfg=json.loads(Path(__file__).with_name('r12_sgd_settings.json').read_text())
    cfg.update(tr.arm_settings('A'))
    model=network.define_model(encoder_model='ResNet12_ReLU',classifier_model='NDCN',way_num=5,
        shot_num=1,use_gpu=False,ndcn_cfg=tr.build_ndcn_cfg(argparse.Namespace(**cfg)))
    optimizer=tr.make_optimizer([{'params':model.parameters(),'weight_decay':.0005}],a)
    before=next(model.features.parameters()).detach().clone()
    model.train()
    logits,aux=model(torch.randn(2,3,84,84),torch.randn(5,3,84,84),torch.tensor([0,1]))
    loss=F.cross_entropy(logits,torch.tensor([0,1]))+.5*aux['total']
    optimizer.zero_grad();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
    assert torch.isfinite(norm)
    optimizer.step()
    assert not torch.equal(before,next(model.features.parameters()))
    print('PASS: original NDPNet updates a real ResNet-12 backbone with SGD')
    print('ALL TESTS PASSED (CPU correctness only; no accuracy claim).')


if __name__=='__main__':main()
