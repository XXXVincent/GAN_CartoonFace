import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_divigance(nn.Module):
    def __init__(self):
        super(KL_divigance, self).__init__()

    def forward(self, pred_S, pred_T):
        p = F.softmax(pred_S, dim=1)
        _kl = torch.sum(p*(F.log_softmax(pred_S, dim=1)-F.log_softmax(pred_T,dim=1)), dim=1)
        return torch.mean(_kl)

class JS_inequality_loss(nn.Module):
    def __init__(self):
        super(JS_inequality_loss, self).__init__()

    def KL_divigance(self, p, q):
        p_logit = F.softmax(p, dim=1)
        _kl = torch.sum(p_logit * (F.log_softmax(p, dim=1) - F.log_softmax(q, dim=1)), dim=1)
        return torch.mean(_kl)

    def forward(self, pred_S, pred_T):
        return 0.5*self.KL_divigance(pred_S, (pred_S+pred_T)/2) +  \
               0.5*self.KL_divigance(pred_T, (pred_S+pred_T)/2)

# test case
pred_p = torch.randn(size=(16,128,48,80))
pred_q = torch.randn(size=(16,128,48,80))

a = KL_divigance()
b = a(pred_p, pred_q)
print(b)
