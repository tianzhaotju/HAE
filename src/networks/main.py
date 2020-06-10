from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .object import OBJECT, OBJECT_Autoencoder
from .texture import TEXTURE, TEXTURE_Autoencoder
from .mnist_motivation import MNIST_Motivation,MNIST_Motivation_Autoencoder
from .texture_hae import TEXTURE_HAE, TEXTURE_HAE_Autoencoder
from .object_hae import OBJECT_HAE, OBJECT_HAE_Autoencoder
from .mnist_hae import MNIST_HAE,MNIST_HAE_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'object', 'texture', 'mnist_motivation', 'mnist_hae', 'texture_hae', 'object_hae')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'object':
        net = OBJECT()

    if net_name == 'texture':
        net = TEXTURE()
    if net_name == 'mnist_motivation':
        net = MNIST_Motivation()
    if net_name == 'mnist_hae':
        net = MNIST_HAE()
    if net_name == 'object_hae':
        net = OBJECT_HAE()
    if net_name == 'texture_hae':
        net = TEXTURE_HAE()
    if net_name == 'minist_hae':
        net = MNIST_HAE()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'object', 'texture', 'mnist_motivation', 'mnist_hae', 'texture_hae', 'object_hae')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'object':
        #ae_net = OBJECT_Autoencoder()
        ae_net = OBJECT_HAE()

    if net_name == 'texture':
        ae_net = TEXTURE_Autoencoder()

    if net_name == 'mnist_motivation':
        ae_net = MNIST_Motivation_Autoencoder()

    if net_name == 'mnist_hae':
        ae_net = MNIST_HAE_Autoencoder()
    if net_name == 'object_hae':
        ae_net = OBJECT_HAE()
    if net_name == 'texture_hae':
        ae_net = TEXTURE_HAE()
        # ae_net = TEXTURE_HAE_Autoencoder()

    if net_name == 'mnist_hae':
        ae_net = MNIST_HAE_Autoencoder()

    return ae_net
