from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from utils.pytorch_ssim import pytorch_ssim
from utils.plot_rescons import plot_reconstruction


import logging
import time
import torch
import torch.optim as optim
import numpy as np


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0, dataset_name: str = 'mnist', ae_loss_type: str = 'l2', ae_only: bool = False):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader,dataset_name, ae_loss_type, ae_only)
        self.test_scores = None

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):


            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                if self.dataset_name == 'object' or self.dataset_name =='texture':
                    inputs, _, _ = data
                else:
                    inputs, _, _ = data

                inputs = inputs.to(self.device)

                # np.save('./log/object/'+str(dataset.normal_classes)+'/Images.npy', inputs.cpu().numpy()[-32:])
                # exit(0)
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)

                if self.ae_loss_type == 'ssim':
                    scores = -ssim_loss(inputs, outputs)
                else:
                    scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))

                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average= False)
        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()


        with torch.no_grad():
            for data in test_loader:
                if self.dataset_name == 'object' or self.dataset_name =='texture':
                    inputs, labels, idx = data
                else:
                    inputs, labels, idx = data
                inputs = inputs.to(self.device)
                np.save('./log/object/'+str(dataset.normal_classes)+'/Images.npy', inputs.cpu().numpy()[-32:])
                outputs = ae_net(inputs)
                if self.ae_loss_type == 'ssim':
                    scores = -ssim_loss(inputs,outputs)
                else:
                    scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                if self.dataset_name == 'object' or self.dataset_name == 'texture':
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist()))
                else:
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))
        if self.dataset_name == 'object' or self.dataset_name == 'texture':
            _, labels, scores = zip(*idx_label_score)
        else:
            _, labels, scores = zip(*idx_label_score)


        labels = np.array(labels)
        scores = np.array(scores)

        labels[labels>0]=1

        auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

        test_time = time.time() - start_time
        self.test_scores = idx_label_score
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
        plot_reconstruction(ae_net, self.device, self.ae_loss_type, dataset.normal_classes)

