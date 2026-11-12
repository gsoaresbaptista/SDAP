import torch


@torch.no_grad()
def get_data(model, loader):
    model.eval()
    all_feas, all_target, all_outs, all_indices = [[] for _ in range(4)]

    for inputs, target, indices, *_ in loader:
        feas, outs = model(inputs)
        all_feas.append(feas)
        all_outs.append(outs)
        all_target.append(target)
        all_indices.append(indices)

    all_feas = torch.concat(all_feas)
    all_outs = torch.concat(all_outs)
    all_target = torch.concat(all_target)
    all_indices = torch.concat(all_indices)

    return all_feas, all_target, all_outs, all_indices
