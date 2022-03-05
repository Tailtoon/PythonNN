import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
#from model import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Классы нейронной сети
# Первый реализует дескриминатор, задача которого в том, чтобы отличить
# сгенерированное изображение от реального.
# Второй реализует генератор, задача которого в генерации изображений, получая
# на вход шум
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 1)
        self.activation = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 784)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)


# Гиперпараметры
epochs = 10
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

def main():
    # Обучение производим эпохами, количество которых задается гиперпараметром
    for epoch in range(epochs):
        for idx, (imgs, _) in enumerate(train_loader):
            idx += 1
            # Обучаем дискриминатор
            # real_inputs - изображения из набора данных MNIST 
            # fake_inputs - изображения от генератора
            # real_inputs должны быть классифицированы как 1, а fake_inputs - как 0
            real_inputs = imgs.to(device)
            real_outputs = D(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_inputs = G(noise)
            fake_outputs = D(fake_inputs)
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
            fake_inputs = G(noise)
            fake_outputs = D(fake_inputs)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
            G_loss = loss(fake_outputs, fake_targets)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            if idx % 100 == 0 or idx == len(train_loader):
                print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, idx, D_loss.item(), G_loss.item()))
        if (epoch+1) % 10 == 0:
            torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
            print('Model saved.')
    latent_space_samples = torch.randn(batch_size, 128).to(device=device)
    generated_samples = G(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()

    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()