import imp
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torch.autograd import Variable
#from model import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt
import os

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_channels = in_channels)
    
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
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
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False) 
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
    def forward(self, x): 
        bn = self.bn(self.relu(self.conv(x))) 
        out = self.avg_pool(bn) 
        return out

class DenseNet(nn.Module): 
    def __init__(self, nr_classes): 
        super(DenseNet, self).__init__() 
  
        self.lowconv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 3, bias = False) 
        self.relu = nn.ReLU()
    
        # Make Dense Blocks 
        self.denseblock1 = self._make_dense_block(Dense_Block, 64) 
        self.denseblock2 = self._make_dense_block(Dense_Block, 128)
        self.denseblock3 = self._make_dense_block(Dense_Block, 128)
        # Make transition Layers 
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
        self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)
        # Classifier 
        self.bn = nn.BatchNorm2d(num_features = 64) 
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
        out = self.relu(self.lowconv(x))

        out = self.denseblock1(out) 
        out = self.transitionLayer1(out)

        out = self.denseblock2(out) 
        out = self.transitionLayer2(out) 
    
        out = self.denseblock3(out) 
        out = self.transitionLayer3(out) 
 
        out = self.bn(out) 
        out = out.view(-1, 64*4*4) 
    
        out = self.pre_classifier(out) 
        out = self.classifier(out)
        return out

# Классы нейронной сети
# Первый реализует дескриминатор, задача которого в том, чтобы отличить
# сгенерированное изображение от реального.
# Второй реализует генератор, задача которого в генерации изображений, получая
# на вход шум
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.fc1 = nn.Linear(794, 512)
        self.fc2 = nn.Linear(512, 1)
        self.activation = nn.LeakyReLU(0.1)
    def forward(self, x, labels):
        x = x.view(-1, 784)
        y = self.label_emb(labels)
        x = torch.cat([x, y], 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.fc1 = nn.Linear(138, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 784)
        self.activation = nn.ReLU()
    def forward(self, x, labels):
        x.view(-1, 128)
        y = self.label_emb(labels)
        x = torch.cat([x, y], 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)


# Гиперпараметры
epochs = 100
lr = 2e-4
batch_size = 32
# Логистическая функция потерь (бинарная перекрестная энтропия) 
loss = nn.BCELoss()

# Создание моделей и передача их на устройство (в нашем случае - это видеокарта)
G = generator().to(device)
D = discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Создание преобразователя изображений из набора MNIST
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Создаем итерируемый загрузчик данных из MNIST
train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

def train():
    # Обучение производим эпохами, количество которых задается гиперпараметром
        for epoch in range(epochs):
            for idx, (imgs, labels) in enumerate(train_loader):
                labels = labels.to(device)
                idx += 1
                # Обучаем дискриминатор
                # real_inputs - изображения из набора данных MNIST 
                # fake_inputs - изображения от генератора
                # real_inputs должны быть классифицированы как 1, а fake_inputs - как 0
                real_inputs = imgs.to(device)
                real_outputs = D(real_inputs, labels)
                real_label = torch.ones(real_inputs.shape[0], 1).to(device)
                noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
                noise = noise.to(device)

                fake_label_to_generate = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)

                fake_inputs = G(noise, fake_label_to_generate)
                fake_outputs = D(fake_inputs, fake_label_to_generate)
                fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)
                outputs = torch.cat((real_outputs, fake_outputs), 0)
                targets = torch.cat((real_label, fake_label), 0)
                D_loss = loss(outputs, targets)
                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()
                # Обучаем генератор
                # Цель генератора получить от дискриминатора 1 по всем изображениям
                noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
                noise = noise.to(device)

                fake_label_to_generate = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)

                fake_inputs = G(noise, fake_label_to_generate)
                fake_outputs = D(fake_inputs, fake_label_to_generate)
                fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
                G_loss = loss(fake_outputs, fake_targets)
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                if idx % 100 == 0 or idx == len(train_loader):
                    print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, idx, D_loss.item(), G_loss.item()))
        torch.save(G, 'Generator.pth')
        print('Model saved.')

def main():
    # lab = torch.LongTensor(np.random.randint(0, 10, batch_size))
    # label_emb = nn.Embedding(10, 10)
    # print(lab.size())
    # print(label_emb(lab).size())
    # print(lab)
    # print(label_emb(lab))
    if os.path.isfile("Generator.pth"):
        try:
            G = torch.load('Generator.pth', map_location=device)
            samples = torch.randn(batch_size, 128).to(device=device)
            noise = (torch.rand(batch_size, 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_label_to_generate = torch.LongTensor([5 for i in range(32)]).to(device)
            generated_samples = G(samples, fake_label_to_generate)
            generated_samples = generated_samples.cpu().detach()
            for i in range(25):
                ax = plt.subplot(5, 5, i + 1)
                plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                plt.xticks([])
                plt.yticks([])
            plt.show()
        except:
            train()
    else:
        train()


if __name__ == "__main__":
    main()