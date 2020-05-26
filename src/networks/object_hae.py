import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class OBJECT_HAE(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.cate_dense_1 = 8
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv0 = nn.Conv2d(3, 32, 8, stride=2, padding=3, bias=True)
        self.bn0 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv1 = nn.Conv2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv6 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv7 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv8 = nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv9 = nn.Conv2d(32, self.rep_dim, 8, stride=1, padding=0, bias=True)
        self.bn9 = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)

        # Representation Layer
        self.dense1 = nn.Linear(self.rep_dim, self.cate_dense_1, bias=True)
        self.dedense1 = nn.Linear(self.cate_dense_1, self.rep_dim, bias=True)
        # Decoder

        self.deconv0 = nn.ConvTranspose2d(32, 3, 8, stride=2, padding=3, bias=True)
        self.debn0 = nn.BatchNorm2d(3, eps=1e-04, affine=False)

        self.deconv1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.debn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.debn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv3 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1, bias=True)
        self.debn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True)
        self.debn4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv5 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.debn5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv6 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True)
        self.debn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv7 = nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.debn7 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.deconv8 = nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1, bias=True)
        self.debn8 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv9 = nn.ConvTranspose2d(self.rep_dim, 32, 8, stride=1, padding=0, bias=True)
        self.debn9 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

    def forward(self, input_x):
        x = self.conv0(input_x)
        # x = F.leaky_relu(self.bn0(x))
        rep_0 = F.leaky_relu(x)
        x = self.deconv0(rep_0)
        # x_reco = F.leaky_relu(self.debn0(x))
        x_reco = torch.sigmoid(x)



        x = self.conv1(rep_0)
        # x = F.leaky_relu(self.bn1(x))
        rep_1 = F.leaky_relu(x)
        x = self.deconv1(rep_1)
        rep_0_reco = F.leaky_relu(x)
        x = self.deconv0(rep_0_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_0_reco = torch.sigmoid(x)



        x = self.conv2(rep_1)
        # x = F.leaky_relu(self.bn2(x))
        rep_2 = F.leaky_relu(x)
        x = self.deconv2(rep_2)
        rep_1_reco = F.leaky_relu(x)
        rep_1_reco = self.deconv1(rep_1_reco)
        #rep_1_reco = F.leaky_relu(rep_1_reco)
        rep_1_reco = self.deconv0(rep_1_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_1_reco = torch.sigmoid(rep_1_reco)



        x = self.conv3(rep_2)
        rep_3 = F.leaky_relu(x)
        x = self.deconv3(rep_3)
        rep_2_reco = F.leaky_relu(x)
        rep_2_reco = self.deconv2(rep_2_reco)
        rep_2_reco = F.leaky_relu(rep_2_reco)
        rep_2_reco = self.deconv1(rep_2_reco)
        rep_2_reco = F.leaky_relu(rep_2_reco)
        rep_2_reco = self.deconv0(rep_2_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_2_reco = torch.sigmoid(rep_2_reco)

        x = self.conv4(rep_3)
        rep_4 = F.leaky_relu(x)
        x = self.deconv4(rep_4)
        rep_3_reco = F.leaky_relu(x)
        rep_3_reco = self.deconv3(rep_3_reco)
        rep_3_reco = F.leaky_relu(rep_3_reco)
        rep_3_reco = self.deconv2(rep_3_reco)
        rep_3_reco = F.leaky_relu(rep_3_reco)
        rep_3_reco = self.deconv1(rep_3_reco)
        rep_3_reco = F.leaky_relu(rep_3_reco)
        rep_3_reco = self.deconv0(rep_3_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_3_reco = torch.sigmoid(rep_3_reco)

        x = self.conv5(rep_4)
        rep_5 = F.leaky_relu(x)
        x = self.deconv5(rep_5)
        rep_4_reco = F.leaky_relu(x)
        rep_4_reco = self.deconv4(rep_4_reco)
        rep_4_reco = F.leaky_relu(rep_4_reco)
        rep_4_reco = self.deconv3(rep_4_reco)
        rep_4_reco = F.leaky_relu(rep_4_reco)
        rep_4_reco = self.deconv2(rep_4_reco)
        rep_4_reco = F.leaky_relu(rep_4_reco)
        rep_4_reco = self.deconv1(rep_4_reco)
        rep_4_reco = F.leaky_relu(rep_4_reco)
        rep_4_reco = self.deconv0(rep_4_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_4_reco = torch.sigmoid(rep_4_reco)

        x = self.conv6(rep_5)
        rep_6 = F.leaky_relu(x)
        x = self.deconv6(rep_6)
        rep_5_reco = F.leaky_relu(x)
        rep_5_reco = self.deconv5(rep_5_reco)
        rep_5_reco = F.leaky_relu(rep_5_reco)
        rep_5_reco = self.deconv4(rep_5_reco)
        rep_5_reco = F.leaky_relu(rep_5_reco)
        rep_5_reco = self.deconv3(rep_5_reco)
        rep_5_reco = F.leaky_relu(rep_5_reco)
        rep_5_reco = self.deconv2(rep_5_reco)
        rep_5_reco = F.leaky_relu(rep_5_reco)
        rep_5_reco = self.deconv1(rep_5_reco)
        rep_5_reco = F.leaky_relu(rep_5_reco)
        rep_5_reco = self.deconv0(rep_5_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_5_reco = torch.sigmoid(rep_5_reco)

        x = self.conv7(rep_6)
        rep_7 = F.leaky_relu(x)
        x = self.deconv7(rep_7)
        rep_6_reco = F.leaky_relu(x)
        rep_6_reco = self.deconv6(rep_6_reco)
        rep_6_reco = F.leaky_relu(rep_6_reco)
        rep_6_reco = self.deconv5(rep_6_reco)
        rep_6_reco = F.leaky_relu(rep_6_reco)
        rep_6_reco = self.deconv4(rep_6_reco)
        rep_6_reco = F.leaky_relu(rep_6_reco)
        rep_6_reco = self.deconv3(rep_6_reco)
        rep_6_reco = F.leaky_relu(rep_6_reco)
        rep_6_reco = self.deconv2(rep_6_reco)
        rep_6_reco = F.leaky_relu(rep_6_reco)
        rep_6_reco = self.deconv1(rep_6_reco)
        rep_6_reco = F.leaky_relu(rep_6_reco)
        rep_6_reco = self.deconv0(rep_6_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_6_reco = torch.sigmoid(rep_6_reco)

        x = self.conv8(rep_7)
        rep_8 = F.leaky_relu(x)
        x = self.deconv8(rep_8)
        rep_7_reco = F.leaky_relu(x)
        rep_7_reco = self.deconv7(rep_7_reco)
        rep_7_reco = F.leaky_relu(rep_7_reco)
        rep_7_reco = self.deconv6(rep_7_reco)
        rep_7_reco = F.leaky_relu(rep_7_reco)
        rep_7_reco = self.deconv5(rep_7_reco)
        rep_7_reco = F.leaky_relu(rep_7_reco)
        rep_7_reco = self.deconv4(rep_7_reco)
        rep_7_reco = F.leaky_relu(rep_7_reco)
        rep_7_reco = self.deconv3(rep_7_reco)
        rep_7_reco = F.leaky_relu(rep_7_reco)
        rep_7_reco = self.deconv2(rep_7_reco)
        rep_7_reco = F.leaky_relu(rep_7_reco)
        rep_7_reco = self.deconv1(rep_7_reco)
        rep_7_reco = F.leaky_relu(rep_7_reco)
        rep_7_reco = self.deconv0(rep_7_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_7_reco = torch.sigmoid(rep_7_reco)


        x = self.conv9(rep_8)
        rep_9 = F.leaky_relu(x)
        x = self.deconv9(rep_9)
        rep_8_reco = F.leaky_relu(x)
        rep_8_reco = self.deconv8(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv7(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv6(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv5(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv4(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv3(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv2(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv1(rep_8_reco)
        rep_8_reco = F.leaky_relu(rep_8_reco)
        rep_8_reco = self.deconv0(rep_8_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_8_reco = torch.sigmoid(rep_8_reco)


        rep_9 = rep_9.view(rep_9.size(0), -1)
        rep_10 = self.dense1(rep_9)
        rep_9_reco = self.dedense1(rep_10)
        rep_9_reco = rep_9_reco.view(rep_9_reco.size(0), self.rep_dim, 1,1)
        rep_9_reco = self.deconv9(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv8(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv7(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv6(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv5(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv4(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv3(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv2(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv1(rep_9_reco)
        rep_9_reco = F.leaky_relu(rep_9_reco)
        rep_9_reco = self.deconv0(rep_9_reco)
        # x_reco = F.leaky_relu(self.debn0(x))
        rep_9_reco = torch.sigmoid(rep_9_reco)
        # exit(0)

        return x_reco, rep_0_reco,  rep_1_reco,  rep_2_reco,  rep_3_reco,  rep_4_reco, \
                rep_5_reco,  rep_6_reco,  rep_7_reco,  rep_8_reco,  rep_9_reco


class OBJECT_HAE_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.cate_dense_1 = 8
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv0 = nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True)
        self.bn0 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv1 = nn.Conv2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv6 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv7 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv8 = nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv9 = nn.Conv2d(32, self.rep_dim, 8, stride=1, padding=0, bias=True)
        self.bn9 = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)

        # Representation Layer
        self.dense1 = nn.Linear(self.rep_dim, self.cate_dense_1, bias=True)
        self.dedense1 = nn.Linear(self.cate_dense_1, self.rep_dim, bias=True)
        # Decoder

        self.deconv0 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True)
        self.debn0 = nn.BatchNorm2d(3, eps=1e-04, affine=False)

        self.deconv1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.debn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.debn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv3 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1, bias=True)
        self.debn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True)
        self.debn4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv5 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.debn5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv6 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True)
        self.debn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv7 = nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.debn7 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.deconv8 = nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1, bias=True)
        self.debn8 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv9 = nn.ConvTranspose2d(self.rep_dim, 32, 8, stride=1, padding=0, bias=True)
        self.debn9 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

    def forward(self, input_x):
        x = self.conv0(input_x)
        # x = F.leaky_relu(self.bn0(x))
        rep_0 = F.leaky_relu(x)
        x = self.deconv0(rep_0)
        # x_reco = F.leaky_relu(self.debn0(x))
        x_reco = F.leaky_relu(x)

        x = self.conv1(rep_0)
        # x = F.leaky_relu(self.bn1(x))
        rep_1 = F.leaky_relu(x)
        x = self.deconv1(rep_1)
        rep_0_reco = F.leaky_relu(x)

        x = self.conv2(rep_1)
        # x = F.leaky_relu(self.bn2(x))
        rep_2 = F.leaky_relu(x)
        x = self.deconv2(rep_2)
        rep_1_reco = F.leaky_relu(x)

        x = self.conv3(rep_2)
        rep_3 = F.leaky_relu(x)
        x = self.deconv3(rep_3)
        rep_2_reco = F.leaky_relu(x)

        x = self.conv4(rep_3)
        rep_4 = F.leaky_relu(x)
        x = self.deconv4(rep_4)
        rep_3_reco = F.leaky_relu(x)

        x = self.conv5(rep_4)
        rep_5 = F.leaky_relu(x)
        x = self.deconv5(rep_5)
        rep_4_reco = F.leaky_relu(x)

        x = self.conv6(rep_5)
        rep_6 = F.leaky_relu(x)
        x = self.deconv6(rep_6)
        rep_5_reco = F.leaky_relu(x)

        x = self.conv7(rep_6)
        rep_7 = F.leaky_relu(x)
        x = self.deconv7(rep_7)
        rep_6_reco = F.leaky_relu(x)

        x = self.conv8(rep_7)
        rep_8 = F.leaky_relu(x)
        x = self.deconv8(rep_8)
        rep_7_reco = F.leaky_relu(x)

        x = self.conv9(rep_8)
        rep_9 = F.leaky_relu(x)
        x = self.deconv9(rep_9)
        rep_8_reco = F.leaky_relu(x)

        rep_9=rep_9.view(rep_9.size(0), -1)
        rep_10 = self.dense1(rep_9)
        rep_9_reco = self.dedense1(rep_10)
        #exit(0)


        return input_x, x_reco, rep_0, rep_0_reco, rep_1, rep_1_reco, rep_2, rep_2_reco, rep_3, rep_3_reco, rep_4, rep_4_reco, \
               rep_5, rep_5_reco, rep_6, rep_6_reco, rep_7, rep_7_reco, rep_8, rep_8_reco, rep_9, rep_9_reco, rep_10


