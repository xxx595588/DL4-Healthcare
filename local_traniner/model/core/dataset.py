import os
from PIL import Image
import torchvision
from torchvision import transforms
import random
random.seed(0)

class CustomDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, is_train=True):
        super(CustomDataset, self).__init__(root, transforms=None)
        self.is_train = is_train
        self.file_lst, self.label_lst = [], []
        for label, folder in enumerate(['no_nCoV', 'nCoV']):
            path = os.path.join(root, folder)
            files = os.listdir(path)
            
            self.file_lst.extend(os.path.join(path, file) for file in files)
            self.label_lst.extend([label] * len(files))

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.126, saturation=0.5) if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        file_path = self.file_lst[index]
        img = Image.open(file_path).convert('RGB')

        img = self.transform(img)
        label = self.label_lst[index]

        return img, label

    def __len__(self):
        return len(self.file_lst)