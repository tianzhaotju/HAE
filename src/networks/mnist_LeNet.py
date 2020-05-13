
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.cate_dense_1 = 10
        self.cate_dense_2 = 10

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=True)

        self.dense1 = nn.Linear(self.rep_dim, self.cate_dense_1, bias=True)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(self.cate_dense_1, self.cate_dense_2, bias=True)
        self.softmax = nn.Softmax()

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=True, padding=2)

    def forward(self, x):
        # # layer 5
        # x = self.conv1(x)
        # x = F.leaky_relu(self.bn1(x))
        # x = self.conv2(x)
        # x = F.leaky_relu(self.bn2(x))
        # x = self.conv3(x)
        # x = F.leaky_relu(self.bn3(x))
        # x = self.conv4(x)
        # x = F.leaky_relu(self.bn4(x))
        # x = self.conv5(x)
        #
        # rep = x.view(x.size(0), -1)
        #
        #
        # rep = self.dense(rep)
        # x = self.deconv5(x)
        # x = self.deconv4(x)
        # x = F.leaky_relu(self.debn4(x))
        # x = self.deconv3(x)
        # x = F.leaky_relu(self.debn3(x))
        # x = self.deconv2(x)
        # x = F.leaky_relu(self.debn2(x))
        # x = self.deconv1(x)
        # x = F.leaky_relu(self.debn1(x))

        # # layer 4
        # x = self.conv1(x)
        # x = F.leaky_relu(self.bn1(x))
        # x = self.conv2(x)
        # x = F.leaky_relu(self.bn2(x))
        # x = self.conv3(x)
        # x = F.leaky_relu(self.bn3(x))
        # x = self.conv4(x)
        # rep = x.view(x.size(0), -1)
        # rep = self.dense(rep)
        # x = self.deconv4(x)
        # x = self.deconv3(x)
        # x = F.leaky_relu(self.debn3(x))
        # x = self.deconv2(x)
        # x = F.leaky_relu(self.debn2(x))
        # x = self.deconv1(x)
        # x = F.leaky_relu(self.debn1(x))

        # layer 3
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.conv3(x)
        rep = x.view(x.size(0), -1)

        rep = self.dense(rep)

        x = self.deconv3(x)
        x = F.leaky_relu(self.debn3(x))
        x = self.deconv2(x)
        x = F.leaky_relu(self.debn2(x))
        x = self.deconv1(x)
        x = F.leaky_relu(self.debn1(x))
        #
        # # layer 2
        # x = self.conv1(x)
        # x = F.leaky_relu(self.bn1(x))
        # x = self.conv2(x)
        # rep = x.view(x.size(0), -1)
        # rep = self.dense(rep)
        # x = self.deconv2(x)
        # x = F.leaky_relu(self.debn2(x))
        # x = self.deconv1(x)
        # x = F.leaky_relu(self.debn1(x))
        #
        # # layer 1
        # x = self.conv1(x)
        # x = F.leaky_relu(self.bn1(x))
        # rep = x.view(x.size(0), -1)
        # rep = self.dense(rep)
        # x = self.deconv1(x)
        # x = F.leaky_relu(self.debn1(x))

        return rep, rep, x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)

        return x
