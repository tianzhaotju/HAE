import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class TEXTURE(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.cate_dense_1 = 10
        self.cate_dense_2 = 10

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
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(self.cate_dense_1, self.cate_dense_2, bias=True)
        self.softmax = nn.Softmax()

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


    def forward(self, x):
        x = self.conv0(x)
        x = self.pool(F.leaky_relu(self.bn0(x)))
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn6(x)))
        x = self.conv7(x)
        x = self.pool(F.leaky_relu(self.bn7(x)))
        x = self.conv8(x)
        x = self.pool(F.leaky_relu(self.bn8(x)))
        x = self.conv9(x)
        x = self.pool(F.leaky_relu(self.bn9(x)))

        rep = x.view(x.size(0), -1)

        cate = self.dense1(rep)
        cate = self.tanh(cate)
        cate = nn.Dropout(p=0.5)(cate)
        cate = self.dense2(cate)
        cate = self.softmax(cate)


        x = self.deconv9(x)
        x = F.leaky_relu(self.debn9(x))
        x = self.deconv8(x)
        x = F.leaky_relu(self.debn8(x))
        x = self.deconv7(x)
        x = F.leaky_relu(self.debn7(x))
        x = self.deconv6(x)
        x = F.leaky_relu(self.debn6(x))
        x = self.deconv5(x)
        x = F.leaky_relu(self.debn5(x))
        x = self.deconv4(x)
        x = F.leaky_relu(self.debn4(x))
        x = self.deconv3(x)
        x = F.leaky_relu(self.debn3(x))
        x = self.deconv2(x)
        x = F.leaky_relu(self.debn2(x))
        x = self.deconv1(x)
        x = F.leaky_relu(self.debn1(x))
        x = self.deconv0(x)
        x = F.leaky_relu(self.debn0(x))

        return rep, cate, x


class TEXTURE_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
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

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool(F.leaky_relu(self.bn0(x)))
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn6(x)))
        x = self.conv7(x)
        x = self.pool(F.leaky_relu(self.bn7(x)))
        x = self.conv8(x)
        x = self.pool(F.leaky_relu(self.bn8(x)))
        x = self.conv9(x)
        x = self.pool(F.leaky_relu(self.bn9(x)))



        x = self.deconv9(x)
        x = F.leaky_relu(self.debn9(x))
        x = self.deconv8(x)
        x = F.leaky_relu(self.debn8(x))
        x = self.deconv7(x)
        x = F.leaky_relu(self.debn7(x))
        x = self.deconv6(x)
        x = F.leaky_relu(self.debn6(x))
        x = self.deconv5(x)
        x = F.leaky_relu(self.debn5(x))
        x = self.deconv4(x)
        x = F.leaky_relu(self.debn4(x))
        x = self.deconv3(x)
        x = F.leaky_relu(self.debn3(x))
        x = self.deconv2(x)
        x = F.leaky_relu(self.debn2(x))
        x = self.deconv1(x)
        x = F.leaky_relu(self.debn1(x))
        x = self.deconv0(x)
        x = F.leaky_relu(self.debn0(x))

        return  x
