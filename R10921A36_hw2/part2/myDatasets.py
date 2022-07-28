

import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 

from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment
from torchvision.transforms import AutoAugmentPolicy
from PIL import Image
import json 
from torchvision.datasets import DatasetFolder

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
train_tfm = transforms.Compose([
            ## TO DO ##
            # You can add some transforms here
            AutoAugment(),
            # AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            transforms.RandomAffine(0, None, (0.7, 1.3)),
            # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
            transforms.ToTensor(),
            
            # experimental normalization for image classification 
            transforms.Normalize(means, stds),
        ])

val_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])


unlabel_tfm = transforms.RandomChoice( 
    [
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1),
            transforms.RandomAffine(0, None, (0.8, 1.2)),
            transforms.ToTensor(),
        ]),
        
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.RandomAffine(0, None, (0.8, 1.2)),
            transforms.ToTensor(),
        ])
    ]
)
class PsudeoDataset(Dataset):
    def __init__(self, dataset, labels, tfm=unlabel_tfm):
        self.data = dataset
        self.labels = labels
        self.tfm = tfm

    def __getitem__(self, idx):
        img = self.data[idx][0]
        img = transforms.ToPILImage()(img).convert("RGB")
        return self.tfm(img), self.labels[idx]

    def __len__(self):
        return len(self.labels)

def get_cifar10_train_val_set(root, ratio=0.9, cv=0):
    
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']

    
    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]
    

    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    train_transform = train_tfm
  
    # normally, we dont apply transform to test_set or val_set
    val_transform = val_tfm

 
  
    ## TO DO ##
    # Complete class cifiar10_dataset
    train_set, val_set = cifar10_dataset(images=train_image, labels=train_label,transform=train_transform, prefix = './p2_data/train'), \
                        cifar10_dataset(images=val_image, labels=val_label,transform=val_transform, prefix = './p2_data/train')


    return train_set, val_set




def get_cifar10_unlabeled_set(root, ratio=0.9, cv=0):
    # return class cifar10_unlabeled_dataset(Dataset):
    # return DatasetFolder(root, loader=lambda x: Image.open(x), extensions="jpg", transform=unlabel_tfm)
    images = []
        
    files = os.listdir(root)

    for file in files:
        # make sure file is an image
        if file.endswith(('.jpg', '.png', 'jpeg')):
            # img_path = self.prefix + file
            images.append(file)
        
        
    unlabeled_set = cifar10_dataset(images=images,transform=unlabel_tfm, prefix = root)


    return unlabeled_set


## TO DO ##
# Define your own cifar_10 dataset
class cifar10_dataset(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        print('from', self.prefix)
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        
        # You shall return image, label with type "long tensor" if it's training set
        # pass
        full_path = os.path.join(self.prefix, self.images[idx])
        img = Image.open(full_path).convert("RGB")
        transform_img = self.transform(img)
        if self.labels != None:
            #  print(type((transform_img, self.labels[idx])))
             return (transform_img, self.labels[idx])
        else:
             return (transform_img)