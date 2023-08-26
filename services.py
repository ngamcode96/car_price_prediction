def load_data(json_file):
    import json
    with open(json_file, "rb") as json_data:
        data = json.load(json_data)
    return data