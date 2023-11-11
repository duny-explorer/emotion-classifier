from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np


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

            with open('data/label_encoder.pkl', 'wb') as le_dump_file:
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


def get_datasets(batch_size=128, balanced=True):
    TRAIN_DIR = Path('data/train/simpsons_dataset')
    TEST_DIR = Path('data/testset/testset')

    train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
    test_files = sorted(list(TEST_DIR.rglob('*.jpg')))

    train_val_labels = [path.parent.name for path in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, \
                                          stratify=train_val_labels)

    val_dataset = SimpsonsDataset(val_files, mode='val')    
    train_dataset = SimpsonsDataset(train_files, mode='train')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if balanced:
        count_weights = {k: 1/v for k, v in counts.items()}
        sample_weights = len(train_files) * [0]

        for i, (data, label) in enumerate(tqdm(train_dataset)):
            label_weight = count_weights[train_dataset.label_encoder.inverse_transform([label])[0]]
            sample_weights[i] = label_weight

        train_sampler = WeightedRandomSampler(sample_weights, num_samples=counts.max() * len(count_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = SimpsonsDataset(test_files, mode="test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
    
    return train_loader, val_loader, test_loader
