import argparse
import json
import pathlib

from models.projector import Projector
from process.train import train
from utils.setup import load_configs_yaml, set_seed
from models.feature_matrix import FEATURE_SIZE


def run_table():
    DATASETS = [
        "office-31",
        "office-home",
        "caltech-imagenet",
    ]
    EPOCHS = {"office-31": 10, "office-home": 30, "caltech-imagenet": 15}
    running = True

    while running:
        try:
            for backbone in FEATURE_SIZE.keys():
                FILE_NAME = f"all_metrics_{backbone}_label_distillation.json"

                if pathlib.Path(FILE_NAME).is_file():
                    with open(FILE_NAME, "r") as old_result:
                        results = json.loads(old_result.read())
                else:
                    results = {}

                for dataset_name in DATASETS:
                    experiment = load_configs_yaml(
                        dataset={"name": dataset_name, "source": "", "target": ""}
                    )
                    experiment['backbone'] = backbone
                    experiment["train"]["epochs"] = EPOCHS[dataset_name]
                    DOMAINS = experiment["dataset"]["domains"]

                    for i in range(5):
                        for source in DOMAINS:
                            for target in DOMAINS:
                                if (source == target) or (
                                    f"{source}_{target}_{i+1}" in results
                                ):
                                    continue

                                experiment["dataset"]["source"] = source
                                experiment["dataset"]["target"] = target

                                print(f"[{i+1}] {source} -> {target}")
                                features = FEATURE_SIZE[experiment['backbone']]
                                model = Projector(
                                    experiment["dataset"]["n_classes"], "small_2", features
                                )
                                steps, single_steps, final_metrics, label_distillation = train(
                                    model, experiment, 0.1, []
                                )
                                results[f"{source}_{target}_{i+1}"] = {
                                    "all_data_until_increment_steps": steps,
                                    "each_step_eval": single_steps,
                                    "re_eval": final_metrics,
                                    # "label_distillation": label_distillation,
                                }
                                with open(FILE_NAME, "w") as json_file:
                                    json.dump(results, json_file, indent=4, sort_keys=True)
        except KeyboardInterrupt:
            exit(0)
        except Exception as error:
            print(error)
            raise error
            continue
        else:
            running = False
            break


def main():
    experiment = load_configs_yaml()
    features = features = FEATURE_SIZE[experiment['backbone']]
    model = Projector(
        experiment["dataset"]["n_classes"], "small_2", features
    )
    # set_seed(42)
    steps, single_step, final_metrics, label_distillation = train(model, experiment, 0.1, [])
    print(steps, single_step, final_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="RAPTOR")
    parser.add_argument("--table", action="store_true", default=False)
    args = parser.parse_args()

    if args.table:
        run_table()
    else:
        main()
