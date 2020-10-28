import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(100, 4 * 4 * 1024),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            nn.Tanh()
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # 入力は(-1, 100)
        out = self.layer1(z)
        out = out.reshape(-1, 1024, 4, 4)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.layer5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2)
            nn.Sigmoid()
        )

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.layer5(out)
        out=out.reshape(-1,1)
        return out


# import torchsummary
def init_weights(m):
    nn.init.normal_(m.weight, 0, 0.02)


# generator = Discriminator()
# # generator.apply(init_weights)
#
# import torchsummary
#
# torchsummary.summary(generator, (3,32,32))
