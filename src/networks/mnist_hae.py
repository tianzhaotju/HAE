import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MNIST_HAE(BaseNet):

    def __init__(self):
        super().__init__()

        # input size [1, 28, 28]
        self.rep_dim1 = int(8 * 24 * 24)
        self.rep_dim2 = int(16 * 12 * 12)
        self.rep_dim3 = int(16 * 6 * 6)
        self.rep_dim4 = int(16 * 3 * 3)
        self.rep_dim5 = int(9 * 1 * 1)

        self.rep_dim_former = self.rep_dim5

        self.rep_dim = 4

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
        self.dedense = nn.Linear(self.rep_dim, self.rep_dim_former, bias=True)

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
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        rep = self.fc1(x)

        cate = self.dense1(rep)
        cate = self.tanh(cate)
        cate = nn.Dropout(p=0.5)(cate)
        cate = self.dense2(cate)
        cate = self.softmax(cate)

        x = rep.view(rep.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)

        return rep, cate, x


class MNIST_HAE_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        # input size [1, 28, 28]
        self.rep_dim1 = int(8 * 24 * 24)
        self.rep_dim2 = int(16 * 12 * 12)
        self.rep_dim3 = int(16 * 6 * 6)
        self.rep_dim4 = int(16 * 3 * 3)
        self.rep_dim5 = int(9 * 1 * 1)

        self.rep_dim_former = self.rep_dim5

        self.rep_dim = 4

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
        self.dedense = nn.Linear(self.rep_dim, self.rep_dim_former, bias=True)

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

    def forward(self, x_input):
        # layer 5
        x = self.conv1(x_input)
        rep_1 = F.leaky_relu(self.bn1(x))
        x = self.deconv1(rep_1)
        input_reco = torch.sigmoid(x)


        x = self.conv2(rep_1)
        rep_2 = F.leaky_relu(self.bn2(x))
        x = self.deconv2(rep_2)
        x = self.deconv1(x)
        rep_1_reco = torch.sigmoid(x)


        x = self.conv3(rep_2)
        rep_3 = F.leaky_relu(self.bn3(x))
        x = self.deconv3(rep_3)
        x = self.deconv2(x)
        x = self.deconv1(x)
        rep_2_reco = torch.sigmoid(x)

        x = self.conv4(rep_3)
        rep_4 = F.leaky_relu(self.bn4(x))
        x = self.deconv4(rep_4)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        rep_3_reco = torch.sigmoid(x)


        x = self.conv5(rep_4)
        rep_5 = F.leaky_relu(self.bn5(x))
        x = self.deconv5(rep_5)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        rep_4_reco = torch.sigmoid(x)


        rep_5 = rep_5.view(rep_5.size(0), -1)
        rep_5 = self.dense(rep_5)
        rep_5 = self.dedense(rep_5)
        rep_5 = rep_5.view(rep_5.size(0), 9, 1,1)
        x = self.deconv5(rep_5)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        rep_5_reco = torch.sigmoid(x)

        return input_reco, rep_1_reco,  rep_2_reco,  rep_3_reco,   rep_4_reco, rep_5_reco
