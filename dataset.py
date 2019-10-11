import torch
import  pandas as pd
from torchvision import transforms
import os
import numpy  as np
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ProductDataset(torch.utils.data.Dataset):
    def __init__(self,label_file,data_path,file_index=None,transform = None):
        label_file = pd.read_csv(label_file)
        if file_index:
            self.file_name=[] 
            self.labels = []
            for index  in file_index:
                self.file_name.append(label_file["FileName"][index])
                self.labels.append(label_file["type"][index])
        else:
            self.file_name=label_file["FileName"]
            self.labels=label_file["type"]
        self.data_path = data_path
        self.transform = transform
    def __len__(self):
        return len(self.file_name)

    def __getitem__(self,index):
        image_path = os.path.join(self.data_path + '/',str( self.file_name[index]))
        image = Image.open(image_path)         
        image = image.convert('RGB')
        label = int(self.labels[index])-1
        if self.transform:
            image = self.transform(image)
        return image,label,index

        
class TestDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,transform=None):
        self.data_path=data_path
        self.file_name= os.listdir(data_path)
#        self.file_name.remove('23110f60ff424b0dbdfa10b78aaad14f.webp')
        self.transform = transform
    def __len__(self):
        return len(self.file_name)


    def __getitem__(self,index):
        image_path = os.path.join(self.data_path + '/',str( self.file_name[index]))
        if str( self.file_name[index]).split('.')[1] != 'webp':
            image = Image.open(image_path)
            image = image.convert('RGB')
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = image = Image.fromarray(np.array(image))
        if self.transform:
            image = self.transform(image)
        return image,str( self.file_name[index])
if __name__ == '__main__':
    val_dataset = TestDataset('test',
                                transform=transforms.Compose([
                                    transforms.Pad(4),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ]))
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=32,
                                                shuffle=False,
                                                **kwargs)

    count = 0
    for i, data in enumerate(test_loader, 0):
        images , labels = data

        print(images.shape,count)
        count += 1
