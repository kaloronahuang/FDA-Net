# FDANet.py
from torch import nn

class FDANetGenerator(nn.Module):

    def __init__(self, embedding_size=512, feature_size=128):
        super(FDANetGenerator, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(embedding_size, feature_size * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size * 16, feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size * 8, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size * 4, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

class FDANetDiscriminator(nn.Module):
    def __init__(self, feature_size=128):
        super(FDANetDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_size * 2, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_size * 4, feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_size * 8, feature_size * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_size * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)