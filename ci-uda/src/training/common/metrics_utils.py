from process.eval import eval
from incremental_old import load_incremental_loader
import torch

@torch.no_grad()
def get_final_eval_metrics(model, experiment, device, n_classes, class_increment):
    """
    Avalia o modelo em cada incremento de classes após o treinamento incremental completo.
    Retorna um dicionário com acurácia e F1-score por incremento.
    """
    model.eval()
    results = {}

    for increment_id in range(n_classes // class_increment):
        _, first_increment, _ = load_incremental_loader(
            experiment, increment_id * class_increment, class_increment, device, False
        )
        acc_increment, f1_increment = eval(model, first_increment, device)
        results[increment_id] = {"acc": acc_increment, "f1": f1_increment}

    return results
