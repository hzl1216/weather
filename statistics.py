
from torchvision import transforms
from dataset import ProductDataset,TestDataset
transforms_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Load data
input_size=224
train_dataset = ProductDataset('Train_label.csv','train',
                                 transform=transforms.Compose([
                                 transforms.RandomRotation(10),
#                                 transforms.Resize([resize_shape,resize_shape]),
#                                 transforms.RandomResizedCrop(input_size ),
                                 transforms.RandomResizedCrop(input_size),
                                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms_norm
                                 ]))