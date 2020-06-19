from .main import build_network, build_autoencoder
from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .object import OBJECT, OBJECT_Autoencoder
from .texture import TEXTURE, TEXTURE_Autoencoder
from .mnist_motivation import MNIST_Motivation,MNIST_Motivation_Autoencoder
from .texture_hae import TEXTURE_HAE, TEXTURE_HAE_Autoencoder
from .object_hae import OBJECT_HAE, OBJECT_HAE_Autoencoder
from .mnist_hae import MNIST_HAE,MNIST_HAE_Autoencoder
