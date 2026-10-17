#!/usr/bin/env python3
import argparse
import torch
from torch.nn import functional as F
from paired_common import make_model,seed_all,state_hash,audit_data,mask_stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--data');a=p.parse_args()
    torch.set_num_threads(2)
    hashes=[]
    for arm in ('ndm','uniform'):
        seed_all(0);m=make_model({'encoder':'ResNet12_ReLU','shot':1},arm,'cpu')
        hashes.append(state_hash(m.state_dict()))
        counts=sum(p.numel() for p in m.parameters())
        print(arm,'registered parameters=',counts,'M=',counts/1e6)
        q=torch.rand(5,3,84,84);s=torch.rand(5,3,84,84);y=torch.arange(5)
        logits,aux=m(q,s,y);loss=F.cross_entropy(logits,y)+.5*aux['total'];loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if arm=='uniform':
            m.eval();m.classifier.record_diagnostics=True
            with torch.no_grad():m(q,s)
            torch.testing.assert_close(m.classifier._last_Mq,torch.ones_like(m.classifier._last_Mq))
        print('PASS:',arm,'forward/backward')
    assert hashes[0]==hashes[1]
    assert abs(mask_stats(torch.ones(2,25))[3]-25)<1e-8
    print('PASS: identical initialization and uniform spatial diagnostics')
    if a.data:print(audit_data(a.data,5,full=True))


if __name__=='__main__':main()
