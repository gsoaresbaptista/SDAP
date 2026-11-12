import torch
import torch.nn.functional as F


def nce_criterion(query, pos, negs, temperature=0.07):
    """
    Noise Contrastive Estimation (NCE) loss using einsum.

    Args:
        query (torch.Tensor): Tensor with query vectors (shape: [batch_size, feature_dim]).
        pos (torch.Tensor): Tensor with positive vectors (shape: [batch_size, feature_dim]).
        negs (torch.Tensor): Tensor with negative vectors (shape: [batch_size, num_negatives, feature_dim]).
        temperature (float): Temperature for scaling logits in softmax.

    Returns:
        torch.Tensor: Scalar NCE loss.
    """
    pos_sim = torch.einsum("bd,bd->b", query, pos).unsqueeze(
        1
    )  # Shape: [batch_size, 1]
    negs_sim = torch.einsum(
        "bd,bnd->bn", query, negs
    )  # Shape: [batch_size, num_negatives]
    logits = torch.cat(
        [pos_sim, negs_sim], dim=1
    )  # Shape: [batch_size, 1 + num_negatives]
    logits /= temperature
    # positives are always at index 0
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=query.device)
    loss = F.cross_entropy(logits, labels)

    return loss


if __name__ == "__main__":
    ...
    # pos = source_center[s_labels]  # Shape: (batch_size, feature_dim)
    # neg_mask = torch.arange(source_center.size(0), device=device).unsqueeze(0) != s_labels.unsqueeze(1)  # Shape: (batch_size, num_classes)
    # negs = source_center.unsqueeze(0).expand(s_labels.size(0), -1, -1)[neg_mask]  # Shape: (batch_size * (n_classes-1), feature_dim)
    # negs = negs.view(s_labels.size(0), -1, source_center.size(1))  # Shape: (batch_size, n_classes-1, feature_dim)
    # nce_loss = nce_criterion(s_features, pos, negs, 0.07)
    # nce_loss = nce_loss.mean()
