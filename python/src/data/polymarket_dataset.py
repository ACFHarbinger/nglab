"""
Polymarket Dataset for time series training.
"""
import os
import torch

from .data_utils import read_json, read_csv, df_to_torch


class PolymarketDataset(torch.utils.data.Dataset):
    """
    Dataset class for Polymarket prediction data.
    """
    def __init__(self, name, dataset_dir, seq_len, pred_len, download=False, transform=None):
        """
        Initialize the dataset.

        Args:
            name (str): Dataset name.
            dataset_dir (str): Directory containing the data.
            seq_len (int): Input sequence length.
            pred_len (int): Prediction sequence length.
            download (bool): Whether to download data (not implemented).
            transform (callable): Optional transform to apply.
        """
        super().__init__()
        if download:
            self._download()
        
        self.name = name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform
        self.dataset_dir = dataset_dir

        self.metadata = None
        self.dataset = {}
        self.dataset_len = 0
        self.labels = None
        self._load_data()

    def _download(self):
        raise NotImplementedError
    
    def __len__(self):
        """Return the total number of samples."""
        return self.dataset_len

    def _get_name(self):
        return self.name

    def _set_transform(self, transform):
        self.transform = transform

    def _load_data(self, lock=None):
        metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        metadata = read_json(metadata_path, lock)
        sorted(metadata, key=lambda x: x['id'])
        self.dataset['Price'] = torch.full((1, self.seq_len), 0.5)
        self.dataset['Labels'] = torch.full((1, self.pred_len), 0.5)
        for md in metadata:
            filepath = os.path.join(self.dataset_dir, md['filename'])
            df = read_csv(filepath, lock)
            torch_dict = df_to_torch(df, "UTC")
            if len(md['options']) == 2:
                # f"{md['category'][0]}{md['id']}"
                torch_size = torch_dict['Price'].size()[0]
                size_diff = (self.seq_len + self.pred_len) - torch_size % (self.seq_len + self.pred_len)
                new_tensor = torch_dict['Price']
                if size_diff > 0:
                    pad_value = round(float(torch_dict['Price'][-1]))
                    new_tensor = torch.nn.ConstantPad1d((0, size_diff), pad_value)(new_tensor)

                reshaped = new_tensor.view(-1, self.seq_len + self.pred_len)
                self.dataset['Price'] = torch.cat((self.dataset['Price'], reshaped[:, :self.seq_len]))
                self.dataset['Labels'] = torch.cat((self.dataset['Labels'], reshaped[:, self.seq_len:]))
        self.dataset['Price'] = self.dataset['Price'].unsqueeze(-1)
        #self.dataset['Labels'] = self.dataset['Labels'].unsqueeze(-1)
        self.dataset_len = self.dataset['Price'].size()[0]

    def __getitem__(self, index):
        """
        Get a sample by index.
        """
        data = dict.fromkeys(self.dataset.keys())
        for key in data.keys():
            data[key] = self.dataset[key][index].type(torch.cuda.FloatTensor)

        if self.labels is not None:
            labels = self.labels[index]
        else:
            labels = None

        if self.transform is not None:
            data = self.transform(data)

        return data