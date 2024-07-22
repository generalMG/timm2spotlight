import os
from PIL import Image
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, dataframe, dataset_path, transform=None):
        self.dataframe = dataframe
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.dataframe.iloc[idx]['filepaths'])
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError: # handles missing images in dataset
            return None, None, None
        if self.transform:
            image = self.transform(image)
        label = self.dataframe.iloc[idx]['labels']
        return image, label, image_path