import torch.utils.data as data
import torch
import os
import glob
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
random.seed(10101)

# sub_batch_size=8 默认一个batch的channels就是8 * 4 * 3,concat的形式组合image
# label: 0: real, 1:fake


def make_dataset(real_source_path, real_target_path, fake_source_path, fake_target_path):
    real_source_dataset = []
    real_target_dataset = []
    fake_source_dataset = []
    fake_target_dataset = []
    for filename in glob.glob(os.path.join(real_source_path, '*.jpg')):
        real_source_dataset.append(filename)
    for filename in glob.glob(os.path.join(real_target_path, '*.jpg')):
        real_target_dataset.append(filename)
    for filename in glob.glob(os.path.join(fake_source_path, '*.jpg')):
        fake_source_dataset.append(filename)
    for filename in glob.glob(os.path.join(fake_target_path, '*.jpg')):
        fake_target_dataset.append(filename)
    random.shuffle(real_source_dataset)
    random.shuffle(real_target_dataset)
    random.shuffle(fake_source_dataset)
    random.shuffle(fake_target_dataset)
    return [real_source_dataset, real_target_dataset, fake_source_dataset, fake_target_dataset]


class mytraindata(data.Dataset):
    def __init__(self, real_source_path, real_target_path, fake_source_path, fake_target_path, transform=True, rescale=224):
        super(mytraindata, self).__init__()
        self.train_set_path = make_dataset(real_source_path, real_target_path, fake_source_path, fake_target_path)
        self.rescale = rescale
        self.transform = transform

    def __getitem__(self, item):
        real_source = Image.open(self.train_set_path[0][item])
        real_target = Image.open(random.sample(self.train_set_path[1], 1)[0])
        fake_source = Image.open(random.sample(self.train_set_path[2], 1)[0])
        fake_target = Image.open(random.sample(self.train_set_path[3], 1)[0])
        label = np.array([0, 0, 1, 1])
        label = torch.from_numpy(label)
        transform = transforms.ToTensor()
        if self.rescale:
            real_source = real_source.resize((self.rescale, self.rescale))
            real_target = real_target.resize((self.rescale, self.rescale))
            fake_source = fake_source.resize((self.rescale, self.rescale))
            fake_target = fake_target.resize((self.rescale, self.rescale))
        if self.transform:
            real_source = transform(real_source)
            real_target = transform(real_target)
            fake_source = transform(fake_source)
            fake_target = transform(fake_target)
        image = torch.cat((real_source, real_target, fake_source, fake_target), 0)
        return image, label

    def __len__(self):
        return len(self.train_set_path[0])
