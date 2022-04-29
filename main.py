from ast import Global
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import transforms
from torch.autograd import Variable
from skimage import io, color
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = in_channels)
    
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 12, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
    # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
   
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
   
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
   
        return c5_dense

class Transition_Layer(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__() 
    
        self.relu = nn.ReLU(inplace = True) 
        self.bn = nn.BatchNorm2d(num_features = out_channels) 
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False) 
        #self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
    def forward(self, x): 
        out = self.bn(self.relu(self.conv(x))) 
        #out = self.avg_pool(bn) 
        return out

class UpTransition_Layer(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(UpTransition_Layer, self).__init__() 
    
        self.relu = nn.ReLU(inplace = True) 
        self.bn = nn.BatchNorm2d(num_features = out_channels) 
        #self.conv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride = 2, bias = False)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)   
    def forward(self, x): 
        out = self.bn(self.relu(self.conv(x))) 
        return out

class DenseNet(nn.Module): 
    def __init__(self, nr_classes): 
        super(DenseNet, self).__init__() 
  
        self.lowconv = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 7, padding = 3, bias = False) 
        self.relu = nn.ReLU()
    
        # Make Dense Blocks 
        self.denseblock1 = self._make_dense_block(Dense_Block, 8) 
        self.denseblock2 = self._make_dense_block(Dense_Block, 16)
        self.denseblock3 = self._make_dense_block(Dense_Block, 16)
        # Make transition Layers 
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 20, out_channels = 16) 
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 20, out_channels = 16) 
        self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 20, out_channels = 8)
        # Upsampling transition layers
        self.uptransitionLayer1 = self._make_transition_layer(UpTransition_Layer, in_channels = 8, out_channels = 16) 
        self.uptransitionLayer2 = self._make_transition_layer(UpTransition_Layer, in_channels = 16, out_channels = 20) 
        self.uptransitionLayer3 = self._make_transition_layer(UpTransition_Layer, in_channels = 20, out_channels = 3)
        # Classifier
        self.sigm = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(num_features = 8) 
        self.pre_classifier = nn.Linear(64*4*4, 512) 
        self.classifier = nn.Linear(512, nr_classes)
 
    def _make_dense_block(self, block, in_channels): 
        layers = [] 
        layers.append(block(in_channels)) 
        return nn.Sequential(*layers)
    def _make_transition_layer(self, layer, in_channels, out_channels): 
        modules = [] 
        modules.append(layer(in_channels, out_channels)) 
        return nn.Sequential(*modules)
    def forward(self, x):
        #print(x.shape) 
        out = self.relu(self.lowconv(x))
        #print(out.shape)

        out = self.denseblock1(out)
        #print(out.shape) 
        out = self.transitionLayer1(out)
        #print(out.shape)

        out = self.denseblock2(out) 
        #print(out.shape)
        out = self.transitionLayer2(out)
        #print(out.shape) 
    
        out = self.denseblock3(out) 
        #print(out.shape)
        out = self.transitionLayer3(out) 
        #print(out.shape)
 
        out = self.bn(out)
        #print(out.shape)

        out = self.uptransitionLayer1(out)
        #print(out.shape)
        out = self.uptransitionLayer2(out)
        #print(out.shape)
        out = self.uptransitionLayer3(out)
        #print(out.shape)
        # out = out.view(-1, 64*4*4) 
    
        # out = self.pre_classifier(out) 
        # out = self.classifier(out)
        return self.sigm(out)

class ResImageDataset(Dataset):
    def __init__(self, dir_name, transform=None):
        self.dir_name = dir_name
        self.raw = os.listdir(os.path.join(dir_name, "raw"))
        self.inpainted = os.listdir(os.path.join(dir_name, "inpainted"))
        print("raw", self.raw)
        print("inpainted", self.inpainted)
        self.transform = transform
    
    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        raw_image = io.imread(os.path.join(self.dir_name, "raw", self.raw[index]))
        inpainted_image = io.imread(os.path.join(self.dir_name, "inpainted", self.inpainted[index]))

        sample = None

        if len(raw_image.shape) < 3:
            raw_image = color.gray2rgb(raw_image)
        if len(inpainted_image.shape) < 3:
            inpainted_image = color.gray2rgb(inpainted_image)

        if raw_image.shape != inpainted_image.shape:
            print("Image {} size problem".format(index))
            print(raw_image.shape)
            raw_image = resize(raw_image, (inpainted_image.shape[0], inpainted_image.shape[1]))

        if self.transform:
            sample = {"raw": self.transform(raw_image),
                      "inpainted": self.transform(inpainted_image)}
        else:
            sample = {"raw": raw_image,
                      "inpainted": inpainted_image}

        return sample


print("Is cuda possible = ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Гиперпараметры
epochs = 10
lr = 1e-2
batch_size = 1
# Логистическая функция потерь (бинарная перекрестная энтропия) 
loss = nn.BCELoss()

# Создание моделей и передача их на устройство (в нашем случае - видеокарта)
G = DenseNet(6).to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr)

# Создание преобразователя изображений из набора MNIST
transform = transforms.Compose([transforms.ToTensor()])

train_set = ResImageDataset("Images", transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

def test_train():
    for epoch in range(1, epochs + 1):
        # monitor training loss
        train_loss = 0.0

        #Training
        for i, sample in enumerate(train_loader):
            raw = sample["raw"]
            #print(raw.shape)
            shape = raw.shape
            raw = raw.view(1, shape[1], shape[2], shape[3]).to(device, dtype = torch.float)
            inpainted = sample["inpainted"]
            shape = inpainted.shape
            inpainted = inpainted.view(1, shape[1], shape[2], shape[3]).to(device, dtype = torch.float)
            G_optimizer.zero_grad()
            outputs = G(raw)
            cur_loss = loss(outputs, inpainted)
            cur_loss.backward()
            G_optimizer.step()
            train_loss += cur_loss.item()
            
            train_loss = train_loss/len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    torch.save(G, 'Generator.pth')
    print('Model saved.')

def main():
    global G
    if os.path.isfile("Generator.pth"):
        try:
            G = torch.load('Generator.pth', map_location=device)

            fig = plt.figure()

            img_index = random.randint(0, len(train_set) - 1)

            img_input = train_set[img_index]["raw"]
            shape = img_input.shape
            img_input = img_input.view(1, shape[0], shape[1], shape[2]).to(device, dtype = torch.float)
            print(img_input.shape)

            img_output = G(img_input)
            transform_tmp = transforms.ToPILImage()
            img_output = img_output.view(3, img_output.shape[2], img_output.shape[3]).cpu().detach()
            img_output = transform_tmp(img_output)
            img_input = img_input.view(3, img_input.shape[2], img_input.shape[3]).cpu().detach()
            img_input = transform_tmp(img_input)

            img_inpainted = train_set[img_index]["inpainted"]
            img_inpainted = img_inpainted.cpu().detach()
            img_inpainted = transform_tmp(img_inpainted)

            print(type(img_output))
            #print(img_output.shape)

            ax = plt.subplot(1, 3, 1)

            plt.tight_layout()
            ax.set_title("Input")
            ax.axis('off')
            plt.imshow(img_input)
            plt.pause(0.001)

            ax = plt.subplot(1, 3, 2)

            ax.set_title("Output")
            ax.axis('off')
            plt.imshow(img_output)
            plt.pause(0.001)

            ax = plt.subplot(1, 3, 3)

            ax.set_title("Inpainted")
            ax.axis('off')
            plt.imshow(img_inpainted)
            plt.pause(0.001)
            plt.show()

        except Exception as ex:
            print(ex)
            return
    else:
        print("Else")
        #dataset = ResImageDataset("Images", transform=transform)
        test_train()


if __name__ == "__main__":
    main()