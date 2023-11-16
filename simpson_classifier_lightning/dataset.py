from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.utils.data import WeightedRandomSampler
from PIL import Image
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


class SimpsonsDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    
    def __init__(self, files, mode):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode

        if self.mode not in ['train', 'val', 'test']:
            print(f"{self.mode} is not correct; correct modes: {['train', 'val', 'test']}")
            raise NameError

        self.len_ = len(self.files)
     
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)
            SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '\\'

            with open(f'{SOURCE_DIR}simpson_classifier\\data\\label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)
                      
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image
  
    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
        
    def _prepare_sample(self, image):
        image = image.resize((224, 224))
        return np.array(image)


class SimpsonDataModule(LightningDataModule):
    def __init__(self, batch_size=128, balanced=False):
        super().__init__()
        self.batch_size = batch_size
        self.balanced = balanced

    def setup(self, stage=None):
        SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '\\'
    
        if stage == "fit" or stage is None:
            TRAIN_DIR = Path(f'{SOURCE_DIR}simpson_classifier\\data\\train\\simpsons_dataset')

            train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))

            train_val_labels = [path.parent.name for path in train_val_files]
            train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels)

            val_dataset = SimpsonsDataset(val_files, mode='val')    
            train_dataset = SimpsonsDataset(train_files, mode='train')

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            if self.balanced:
                counts = pd.Series([path.parent.name for path in train_files]).value_counts()
                count_weights = {k: 1/v for k, v in counts.items()}
                sample_weights = len(train_files) * [0]
                print('Start balancing dataset\n')

                for i, (data, label) in enumerate(tqdm(train_dataset)):
                    label_weight = count_weights[train_dataset.label_encoder.inverse_transform([label])[0]]
                    sample_weights[i] = label_weight

                train_sampler = WeightedRandomSampler(sample_weights, num_samples=counts.max() * len(count_weights), replacement=True)
                self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)
                
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        if stage == "test" or stage is None:
            TEST_DIR = Path(f'{SOURCE_DIR}simpson_classifier\\data\\testset\\testset')
            test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
            test_dataset = SimpsonsDataset(test_files, mode="test")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
