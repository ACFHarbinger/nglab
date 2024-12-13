import os
import json
import torch
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


def _df_to_torch(df, key_avoid, fill_nan=0):
    torch_dict = {}
    if fill_nan is not None:
        df = df.fillna(fill_nan)
        
    for col in df.columns:
        if key_avoid not in col:
            torch_dict[col] = torch.tensor(df[col].values)
    return torch_dict


def load_data(data_dir, lock=None):
    metadata_path = os.path.join(data_dir, "metadata.json")
    metadata = read_json(metadata_path, lock)
    data_dict = {}
    if os.path.basename(data_dir) == 'polymarket':
        sorted(metadata, key=lambda x: x['id'])
        for md in metadata:
            filepath = os.path.join(data_dir, md['filename'])
            df = read_csv(filepath, lock)
            data_dict[f"{md['category'][0]}{md['id']}"] = _df_to_torch(df, "UTC")
    return data_dict, metadata