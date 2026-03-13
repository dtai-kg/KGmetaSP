import json 

def read_id_file(path):

    with open(path, 'r') as file:
        lines = file.readlines()

    id_list = [line.strip().split(" ")[0] for line in lines if '.' not in line.strip().split(" ")[0]]

    return id_list

def append_id_to_file(path, id):

    with open(path , 'a') as file:
        file.write(f"{id}\n")

    return

def save_json(path, dict):

    with open(path, 'w') as file:
        json.dump(dict, file, indent=4)

    return

def read_json(path):
    """Function that reads JSON file"""

    with open(path, 'r') as file:
        data = json.load(file)

    return data