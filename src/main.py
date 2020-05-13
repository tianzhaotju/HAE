import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset
import torchvision.transforms as transforms



################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'object', 'texture']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'object', 'texture', 'mnist_motivation', 'object_hae', 'texture_hae', 'mnist_hae']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary', 'deep-GMM', 'hybrid']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--ae_loss_type', type=str, default='l2',
              help='Specify the type of loss for auto-encoder')
@click.option('--ae_only', type=bool, default=False,
              help='Specify only train auto-encoder')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')


def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, ae_loss_type, ae_only ,normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    mean = [(0.36607818518032215, 0.3528722483374472, 0.3585191239764038),  # 0
            (0.4487305946663354, 0.4487305946663354, 0.4487305946663354),  # 1
            (0.3923340318128373, 0.26295472525674995, 0.22025334692657814),  # 2
            (0.4536255693657713, 0.4682865838881645, 0.4452575836280415),  # 3
            (0.672454086143443, 0.4779993567370712, 0.35007702036667776),  # 4
            (0.5352967021800805, 0.5314880132137422, 0.547828897157147),  # 5
            (0.3267409463643222, 0.41484389522093523, 0.46695618025405883),  # 6
            (0.6926364358307354, 0.662149771557822, 0.6490556404776292),  # 7
            (0.24011281595607017, 0.1769201147939173, 0.17123964257174726),  # 8
            (0.21251877631977975, 0.23440739849813622, 0.2363959074824541),  # 9
            (0.3025230547246622, 0.30300693821061303, 0.32466943588225744),  # 10
            (0.7214971293922232, 0.7214971293922232, 0.7214971293922232),  # 11
            (0.20453672401964704, 0.19061953742573437, 0.1973630989492544),  # 12
            (0.38709726938081024, 0.27680750921869235, 0.24161576675737736),  # 13
            (0.39719792798156195, 0.39719792798156195, 0.39719792798156195),  # 14
            ]
    std = [(0.1334089197933497, 0.13091438558839882, 0.11854704285817017),  # 0
           (0.16192189716258867, 0.16192189716258867, 0.16192189716258867),  # 1
           (0.0527090063203568, 0.035927180158353854, 0.026535684323885065),  # 2
           (0.11774565267141425, 0.13039328961987165, 0.12533147519872007),  # 3
           (0.07714836895006975, 0.06278302787607731, 0.04349760909698915),  # 4
           (0.36582285759516936, 0.3661720233895615, 0.34943018535446296),  # 5
           (0.14995070226373788, 0.2117666336616603, 0.23554648659289779),  # 6
           (0.23612927993223184, 0.25644744015075704, 0.25718179933681784),  # 7
           (0.168789697373752, 0.07563237349131141, 0.043146545992581754),  # 8
           (0.15779873915363898, 0.18099161937329614, 0.15159372072430388),  # 9
           (0.15720102988319967, 0.1803989691876269, 0.15113407058442763),  # 10
           (0.13265686578689692, 0.13265686578689692, 0.13265686578689692),  # 11
           (0.2316392849251032, 0.21810285502082638, 0.19743939091294657),  # 12
           (0.20497542590257026, 0.14190994609091834, 0.11531548927488476),  # 13
           (0.3185215984033291, 0.3185215984033291, 0.3185215984033291),  # 14
           ]

    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    xp_path = xp_path + '/'

    xp_path = xp_path + str(normal_class)
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %s' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)

        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader,
                           dataset_name = dataset_name,
                           ae_loss_type = ae_loss_type,
                           ae_only = ae_only)

        # Plot most anomalous and most normal (within-class) test samples
        exit(0)
        indices, labels, scores = zip(*deep_SVDD.results['ae_test_scores'])
        indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
        idx_sorted = indices[labels == 0][
            np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

        if dataset_name in ('mnist', 'cifar10', 'object', 'texture'):

            if dataset_name == 'mnist':
                X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
                X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

            if dataset_name == 'cifar10':
                X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
                X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

            if dataset_name == 'object':
                # 22 3 256 256
                X_normals = torch.tensor(dataset.test_data[idx_sorted[:32], ...])
                X_outliers = torch.tensor(dataset.test_data[idx_sorted[-32:], ...])

                for i in range(3):
                    X_normals[:, i, :, :] *= std[normal_class][i]
                    X_normals[:, i, :, :] += mean[normal_class][i]
                    X_outliers[:, i, :, :] *= std[normal_class][i]
                    X_outliers[:, i, :, :] += mean[normal_class][i]

            #plot_images_grid(X_normals, export_img=xp_path + '/AE_normals', title='Most normal examples', padding=2)
            #plot_images_grid(X_outliers, export_img=xp_path + '/AE_outliers', title='Most anomalous examples', padding=2)
            if ae_only:
                exit(0)
    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    dataset_name = dataset_name)

    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)


    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores, _, _ = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    if dataset_name in ('mnist', 'cifar10', 'object', 'texture'):

        if dataset_name == 'mnist':
            X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
            X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

        if dataset_name == 'object':
            # 22 3 256 256
            X_normals = torch.tensor(dataset.test_data[idx_sorted[:32], ...])
            X_outliers = torch.tensor(dataset.test_data[idx_sorted[-32:], ...])
            for i in range(3):
                X_normals[:, i, :, :] *= std[normal_class][i]
                X_normals[:, i, :, :] += mean[normal_class][i]
                X_outliers[:, i, :, :] *= std[normal_class][i]
                X_outliers[:, i, :, :] += mean[normal_class][i]

        plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
        plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)

    # Save results, model, and configuration
    #deep_SVDD.save_results(export_json=xp_path + '/results.json')
    #deep_SVDD.save_model(export_model=xp_path + '/model.tar')
    #cfg.save_config(export_json=xp_path + '/config.json')

    #
if __name__ == '__main__':
    main()
#python main.py mnist mnist_LeNet ../log/mnist ../data --objective deep-GMM --lr 0.0001 --n_epochs 5 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.001 --ae_n_epochs 10 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class [2,3]
#python main.py cifar10 cifar10_LeNet_ELU ../log/cifar10 ../data --objective deep-GMM --lr 0.0001 --n_epochs 5 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --seed -1 --ae_lr 0.0001 --ae_n_epochs 10 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class [0,1]
#python src/main.py object object ./log/object ./data/Mvtec/ --objective deep-GMM --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --seed -1 --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --ae_loss_type 'ssim' --ae_only True --normal_class 9
#conda activate Deep_GMM

# Motivaton on Mnist

# Auto-encoder
#python src/main.py mnist mnist_motivation ./log/mnist ./data --objective one-class --lr 0.001 --n_epochs 30 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.001 --ae_n_epochs 30 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --ae_only True --normal_class 0
# One-class
#python src/main.py mnist mnist_motivation ./log/mnist ./data --objective one-class --lr 0.001 --n_epochs 50 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 30 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3
#Hybrid
#python src/main.py mnist mnist_motivation ./log/mnist ./data --objective hybrid --lr 0.001 --n_epochs 50 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 30 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3


#HAE script
# 5-14
#python src/main.py object object_hae ./log/object ./data/Mvtec/ --objective deep-GMM --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --seed -1 --ae_lr 0.0001 --ae_n_epochs 30 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --ae_loss_type 'object_HAE' --ae_only True --normal_class 11
# 0-4
#python src/main.py texture texture_hae ./log/texture ./data/Mvtec/ --objective deep-GMM --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --seed -1 --ae_lr 0.0001 --ae_n_epochs 1 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --ae_loss_type 'texture_HAE' --ae_only True --normal_class 0

#python src/main.py mnist mnist_hae ./log/mnist ./data --objective deep-GMM --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --seed -1 --ae_lr 0.0001 --ae_n_epochs 30 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --ae_loss_type 'mnist_HAE' --ae_only True --normal_class 1