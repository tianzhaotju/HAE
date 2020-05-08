import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MNIST_Motivation(BaseNet):

    def __init__(self):
        super().__init__()

        # input size [1, 28, 28]
        self.rep_dim1 = int(8 * 24 * 24)
        self.rep_dim2 = int(16 * 12 * 12)
        self.rep_dim3 = int(16 * 6 * 6)
        self.rep_dim4 = int(16 * 3 * 3)
        self.rep_dim5 = int(9 * 1 * 1)

        self.rep_dim_former = self.rep_dim5
        self.rep_dim = int(self.rep_dim_former / 9)

        self.cate_dense_2 =32

        # Output size [8, 24, 24]
        self.conv1 = nn.Conv2d(1, 8, 5, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        # Output size [16, 12, 12]
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # Output size [16, 6, 6]
        self.conv3 = nn.Conv2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # Output size [16, 3, 3]
        self.conv4 = nn.Conv2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # Output size [8, 1, 1]
        self.conv5 = nn.Conv2d(16, 9, 3, stride=1, padding=0, bias=True)
        self.bn5 = nn.BatchNorm2d(9, eps=1e-04, affine=False)

        self.dense = nn.Linear(self.rep_dim_former, self.rep_dim, bias=True)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(8, 1, 5, stride=1, padding=0, bias=True)
        self.debn1 = nn.BatchNorm2d(1, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=True)
        self.debn2 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.debn3 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.debn4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)

        self.deconv5 = nn.ConvTranspose2d(9, 16, 3, stride=1, padding=0, bias=True)
        self.debn5 = nn.BatchNorm2d(16, eps=1e-04, affine=False)

    def forward(self, x):

        # layer 5
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.conv5(x)

        rep = x.view(x.size(0), -1)


        rep = self.dense(rep)
        x = self.deconv5(x)
        x = self.deconv4(x)
        x = F.leaky_relu(self.debn4(x))
        x = self.deconv3(x)
        x = F.leaky_relu(self.debn3(x))
        x = self.deconv2(x)
        x = F.leaky_relu(self.debn2(x))
        x = self.deconv1(x)
        x = F.leaky_relu(self.debn1(x))


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

        # #layer 3
        # x = self.conv1(x)
        # x = F.leaky_relu(self.bn1(x))
        # x = self.conv2(x)
        # x = F.leaky_relu(self.bn2(x))
        # x = self.conv3(x)
        # rep = x.view(x.size(0), -1)
        #
        # rep = self.dense(rep)
        #
        # x = self.deconv3(x)
        # x = F.leaky_relu(self.debn3(x))
        # x = self.deconv2(x)
        # x = F.leaky_relu(self.debn2(x))
        # x = self.deconv1(x)
        # x = F.leaky_relu(self.debn1(x))
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


class MNIST_Motivation_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        # input size [1, 28, 28]
        self.rep_dim1 = int(8 * 24 * 24)
        self.rep_dim2 = int(16 * 12 * 12)
        self.rep_dim3 = int(16 * 6 * 6)
        self.rep_dim4 = int(16 * 3 * 3)
        self.rep_dim5 = int(9 * 1 * 1)

        self.rep_dim_former = self.rep_dim5

        self.rep_dim = int(self.rep_dim_former / 9)




        # Output size [8, 24, 24]
        self.conv1 = nn.Conv2d(1, 8, 5, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        # Output size [16, 12, 12]
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # Output size [16, 6, 6]
        self.conv3 = nn.Conv2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # Output size [16, 3, 3]
        self.conv4 = nn.Conv2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)

        self.conv5 = nn.Conv2d(16, 9, 3, stride=1, padding=0, bias=True)
        self.bn5 = nn.BatchNorm2d(9, eps=1e-04, affine=False)

        self.dense = nn.Linear(self.rep_dim_former, self.rep_dim, bias=True)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(8, 1, 5, stride=1, padding=0, bias=True)
        self.debn1 = nn.BatchNorm2d(1, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=True)
        self.debn2 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.debn3 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=True)
        self.debn4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(9, 16, 3, stride=1, padding=0, bias=True)
        self.debn5 = nn.BatchNorm2d(16, eps=1e-04, affine=False)

    def forward(self, x):
        # layer 5
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.conv5(x)

        rep = x.view(x.size(0), -1)
        rep = self.dense(rep)

        x = self.deconv5(x)
        x = self.deconv4(x)
        x = F.leaky_relu(self.debn4(x))
        x = self.deconv3(x)
        x = F.leaky_relu(self.debn3(x))
        x = self.deconv2(x)
        x = F.leaky_relu(self.debn2(x))
        x = self.deconv1(x)
        x = F.leaky_relu(self.debn1(x))

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

        ## layer 3
        # x = self.conv1(x)
        # x = F.leaky_relu(self.bn1(x))
        # x = self.conv2(x)
        # x = F.leaky_relu(self.bn2(x))
        # x = self.conv3(x)
        # rep = x.view(x.size(0), -1)
        # rep = self.dense(rep)
        # x = self.deconv3(x)
        # x = F.leaky_relu(self.debn3(x))
        # x = self.deconv2(x)
        # x = F.leaky_relu(self.debn2(x))
        # x = self.deconv1(x)
        # x = F.leaky_relu(self.debn1(x))
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
        return x
