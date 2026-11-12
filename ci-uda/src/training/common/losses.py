import torch
import torch.distributed as dist
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins(z1, z2):
    c = z1.T @ z2
    c.div_(z1.shape[0])
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + 0.0051 * off_diag
    return loss


def vic_reg(x, y):
    repr_loss = F.mse_loss(x, y)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (24 - 1)
    cov_y = (y.T @ y) / (24 - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(2048) + off_diagonal(cov_y).pow_(2).sum().div(
        2048
    )

    loss = 25 * repr_loss + 25 * std_loss + 1 * cov_loss
    return loss


def contrastive_alignment_loss(
    anchor: torch.Tensor, pos_neg_pair: torch.Tensor, gamma: float, margin: float = 0
) -> torch.Tensor:
    result = F.cosine_similarity(anchor.unsqueeze(0), pos_neg_pair, dim=-1)
    max_result = torch.max(result / gamma)
    numerator = torch.exp((result[0] - margin) / gamma - max_result)
    denominator = numerator + torch.sum(torch.exp(result[1:] / gamma - max_result)) + 1e-8
    loss = -torch.log(numerator / denominator)
    return loss


def distillation_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    q = F.softmax(q, dim=1)
    q = torch.log(q)
    soft_pred_loss = torch.sum(-1 * p * q, dim=1)
    return torch.mean(soft_pred_loss)
