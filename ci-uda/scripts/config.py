import json
import os


def generate_config(txt_file_path, output_json_path):
    """
    Generates a config.json file that maps each unique class in a text file to an integer ID.

    Args:
        txt_file_path (str): Path to the text file containing image paths and class IDs.
        output_json_path (str): Path to save the output JSON config file.
    """
    class_to_id = {}

    with open(txt_file_path, 'r') as f:
        for line in f:
            image_path, class_id = line.strip().split()
            class_name = os.path.basename(os.path.dirname(image_path))
            # Store unique class name and ID in the dictionary
            if class_name not in class_to_id:
                class_to_id[class_name] = int(class_id)

    with open(output_json_path, 'w') as json_file:
        json.dump(class_to_id, json_file, indent=4)


if __name__ == '__main__':
    txt_file_path = 'path/to/domain.txt'
    output_json_path = 'path/to/config.json'
    generate_config(txt_file_path, output_json_path)
