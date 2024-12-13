import os
import json
import pandas as pd


def read_json(json_path, lock=None):
    if lock is not None:
        lock.acquire(timeout=10)
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    if lock is not None:
        lock.release()
    return json_data


def read_csv(csv_path, lock=None):
    if lock is not None:
        lock.acquire(timeout=10)
    df = pd.read_csv(csv_path) if os.path.isfile(csv_path) else None
    if lock is not None:
        lock.release()
    return df


def load_data(data_dir):
    metadata_path = os.path.join(data_dir, "metadata.json")
    metadata = read_json(metadata_path)
    data_dict = {}
    filenames = next(os.walk(data_dir), (None, None, []))[2]
    print(filenames)
    return filenames