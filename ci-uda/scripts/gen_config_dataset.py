import json
from pathlib import Path

dataset_path = Path("datasets/caltech_imagenet/caltech")
sorted_classes = sorted(folder.name for folder in dataset_path.glob("*/"))
remap = {name: i for i, name in enumerate(sorted_classes)}

with open("config.json", "w") as f:
    json.dump(remap, f, indent=4)
