from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from utils.pytorch_ssim import pytorch_ssim
from utils.plot_rescons import plot_reconstruction
from sklearn.neighbors import KernelDensity, KDTree


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

                # Update network parameters via backpropagation: forward + backward + optimize
                if self.ae_loss_type == 'object_HAE' or self.ae_loss_type == 'object_HAE_ssim':
                    x_reco, rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = ae_net(inputs)
                if self.ae_loss_type == 'texture_HAE' or self.ae_loss_type == 'texture_HAE_ssim':
                    rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = ae_net(inputs)
                elif self.ae_loss_type == 'mnist_HAE' or self.ae_loss_type == 'mnist_HAE_ssim':
                    x, x_reco, rep_0, rep_0_reco, rep_1, rep_1_reco, rep_2, rep_2_reco, rep_3, rep_3_reco, rep_4= ae_net(inputs)
                else:
                    outputs = ae_net(inputs)

                if self.ae_loss_type == 'object_HAE' :

                    # scores = torch.sum((x - x_reco) ** 2, dim=tuple(range(1, x_reco.dim())))+  \
                    #           torch.sum((rep_0 - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))) + \
                    #           torch.sum((rep_1 - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))) + \
                    #           torch.sum((rep_2 - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))) + \
                    #           torch.sum((rep_3 - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))) + \
                    #           torch.sum((rep_4 - rep_4_reco) ** 2, dim=tuple(range(1, rep_4_reco.dim()))) + \
                    #           torch.sum((rep_5 - rep_5_reco) ** 2, dim=tuple(range(1, rep_5_reco.dim()))) + \
                    #           torch.sum((rep_6 - rep_6_reco) ** 2, dim=tuple(range(1, rep_6_reco.dim()))) + \
                    #           torch.sum((rep_7 - rep_7_reco) ** 2, dim=tuple(range(1, rep_7_reco.dim()))) + \
                    #           torch.sum((rep_8 - rep_8_reco) ** 2, dim=tuple(range(1, rep_8_reco.dim()))) + \
                    #           torch.sum((rep_9 - rep_9_reco) ** 2, dim=tuple(range(1, rep_9_reco.dim()))) + \
                    #           torch.sum((rep_10) ** 2, dim=tuple(range(1, rep_10.dim())))
                    score_nat = torch.sum((inputs - x_reco) ** 2, dim=tuple(range(1, x_reco.dim())))
                    score_0 =torch.sum((x_reco - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim())))
                    score_1 =torch.sum((rep_0_reco - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim())))
                    score_2 =torch.sum((rep_1_reco - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim())))
                    score_3 =torch.sum((rep_2_reco - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim())))
                    score_4 =torch.sum((rep_3_reco - rep_4_reco) ** 2, dim=tuple(range(1, rep_4_reco.dim())))
                    score_5 =torch.sum((rep_4_reco - rep_5_reco) ** 2, dim=tuple(range(1, rep_5_reco.dim())))
                    score_6 =torch.sum((rep_5_reco - rep_6_reco) ** 2, dim=tuple(range(1, rep_6_reco.dim())))
                    score_7 =torch.sum((rep_6_reco - rep_7_reco) ** 2, dim=tuple(range(1, rep_7_reco.dim())))
                    score_8 =torch.sum((rep_7_reco - rep_8_reco) ** 2, dim=tuple(range(1, rep_8_reco.dim())))
                    score_9 =torch.sum((rep_8_reco - rep_9_reco) ** 2, dim=tuple(range(1, rep_9_reco.dim())))
                    if epoch < 0:
                        score_list = [score_nat, score_0, score_1, score_2, score_3, score_4, score_5, score_6, score_7,
                                      score_8, score_9]
                        a = int(self.n_epochs / 12)
                        net_index = int(epoch / a)
                        scores = score_list[net_index]
                        requires_grad_true_list = [[0, 1, 24, 25], [2, 3, 26, 27], [4, 5, 28, 29], [6, 7, 30, 31],
                                                   [8, 9, 32, 33], [10, 11, 34, 35], [12, 13, 36, 37], [14, 15, 38, 39],
                                                   [16, 17, 40, 41], [18, 19, 42, 43], [20, 21, 22, 23]]
                        for i, para in enumerate(ae_net.parameters()):
                            para.requires_grad = False
                            if i in requires_grad_true_list[net_index]:
                                para.requires_grad = True
                        # for i,para in enumerate(ae_net.parameters()):
                        #     print(i,para.requires_grad)

                        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ae_net.parameters()), lr=self.lr,
                                               weight_decay=self.weight_decay,
                                               amsgrad=self.optimizer_name == 'amsgrad')
                    else:
                        for i, para in enumerate(ae_net.parameters()):
                            para.requires_grad = True
                        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr,
                                               weight_decay=self.weight_decay,
                                               amsgrad=self.optimizer_name == 'amsgrad')
                        scores = score_nat+score_0+score_1+ score_2+score_3+score_4+ score_5+score_6+score_7+ score_8+ score_9
                    loss = torch.mean(scores)
                    # Zero the network parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #print(torch.mean(score_nat).item(),torch.mean(score_0).item(),torch.mean(score_1).item(), torch.mean(score_2).item(),torch.mean(score_3).item(),torch.mean(score_4).item(), torch.mean(score_5).item(),torch.mean(score_6).item(),torch.mean(score_7).item(), torch.mean(score_8).item(), torch.mean(score_9).item(),torch.mean(score_10).item())
                elif self.ae_loss_type == 'object_HAE_ssim':
                    score_nat = -ssim_loss(inputs, x_reco)
                    score_0 = -ssim_loss(x_reco, rep_0_reco)
                    score_1 = -ssim_loss(rep_0_reco, rep_1_reco)
                    score_2 = -ssim_loss(rep_1_reco, rep_2_reco)
                    score_3 = -ssim_loss(rep_2_reco, rep_3_reco)
                    score_4 = -ssim_loss(rep_3_reco, rep_4_reco)
                    score_5 = -ssim_loss(rep_4_reco, rep_5_reco)
                    score_6 = -ssim_loss(rep_5_reco, rep_6_reco)
                    score_7 = -ssim_loss(rep_6_reco, rep_7_reco)
                    score_8 = -ssim_loss(rep_7_reco, rep_8_reco)
                    score_9 = -ssim_loss(rep_8_reco, rep_9_reco)
                    scores = score_nat + score_0 + score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8 + score_9
                    loss = torch.mean(scores)
                    # Zero the network parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif self.ae_loss_type == 'texture_HAE':
                    score_0 = torch.sum((inputs - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim())))
                    score_1 = torch.sum((rep_0_reco - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim())))
                    score_2 = torch.sum((rep_1_reco - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim())))
                    score_3 = torch.sum((rep_2_reco - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim())))
                    score_4 = torch.sum((rep_3_reco - rep_4_reco) ** 2, dim=tuple(range(1, rep_4_reco.dim())))
                    score_5 = torch.sum((rep_4_reco - rep_5_reco) ** 2, dim=tuple(range(1, rep_5_reco.dim())))
                    score_6 = torch.sum((rep_5_reco - rep_6_reco) ** 2, dim=tuple(range(1, rep_6_reco.dim())))
                    score_7 = torch.sum((rep_6_reco - rep_7_reco) ** 2, dim=tuple(range(1, rep_7_reco.dim())))
                    score_8 = torch.sum((rep_7_reco - rep_8_reco) ** 2, dim=tuple(range(1, rep_8_reco.dim())))
                    score_9 = torch.sum((rep_8_reco - rep_9_reco) ** 2, dim=tuple(range(1, rep_9_reco.dim())))

                    # score_0 = torch.sum((inputs - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim())))
                    # score_1 = torch.sum((inputs - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim())))
                    # score_2 = torch.sum((inputs - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim())))
                    # score_3 = torch.sum((inputs - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim())))
                    # score_4 = torch.sum((inputs - rep_4_reco) ** 2, dim=tuple(range(1, rep_4_reco.dim())))
                    # score_5 = torch.sum((inputs - rep_5_reco) ** 2, dim=tuple(range(1, rep_5_reco.dim())))
                    # score_6 = torch.sum((inputs - rep_6_reco) ** 2, dim=tuple(range(1, rep_6_reco.dim())))
                    # score_7 = torch.sum((inputs - rep_7_reco) ** 2, dim=tuple(range(1, rep_7_reco.dim())))
                    # score_8 = torch.sum((inputs - rep_8_reco) ** 2, dim=tuple(range(1, rep_8_reco.dim())))
                    # score_9 = torch.sum((inputs - rep_9_reco) ** 2, dim=tuple(range(1, rep_9_reco.dim())))
                    if epoch < 0:
                        score_list = [score_0, score_1, score_2, score_3, score_4, score_5, score_6, score_7,
                                      score_8, score_9]
                        a = int(self.n_epochs / 12)
                        net_index = int(epoch / a)
                        scores = score_list[net_index]
                        requires_grad_true_list = [[0, 1, 24, 25], [2, 3, 26, 27], [4, 5, 28, 29], [6, 7, 30, 31],
                                                   [8, 9, 32, 33], [10, 11, 34, 35], [12, 13, 36, 37], [14, 15, 38, 39],
                                                   [16, 17, 40, 41], [18, 19, 42, 43], [20, 21, 22, 23]]
                        for i, para in enumerate(ae_net.parameters()):
                            para.requires_grad = False
                            if i in requires_grad_true_list[net_index]:
                                para.requires_grad = True
                        # for i,para in enumerate(ae_net.parameters()):
                        #     print(i,para.requires_grad)

                        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ae_net.parameters()), lr=self.lr,
                                               weight_decay=self.weight_decay,
                                               amsgrad=self.optimizer_name == 'amsgrad')
                    else:
                        for i, para in enumerate(ae_net.parameters()):
                            para.requires_grad = True
                        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr,
                                               weight_decay=self.weight_decay,
                                               amsgrad=self.optimizer_name == 'amsgrad')
                        scores = score_0 + score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8 + score_9
                    loss = torch.mean(scores)
                    # Zero the network parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                elif self.ae_loss_type == 'texture_HAE_ssim':

                    score_0 = -ssim_loss(x_reco, rep_0_reco)
                    score_1 = -ssim_loss(rep_0_reco, rep_1_reco)
                    score_2 = -ssim_loss(rep_1_reco, rep_2_reco)
                    score_3 = -ssim_loss(rep_2_reco, rep_3_reco)
                    score_4 = -ssim_loss(rep_3_reco, rep_4_reco)
                    score_5 = -ssim_loss(rep_4_reco, rep_5_reco)
                    score_6 = -ssim_loss(rep_5_reco, rep_6_reco)
                    score_7 = -ssim_loss(rep_6_reco, rep_7_reco)
                    score_8 = -ssim_loss(rep_7_reco, rep_8_reco)
                    score_9 = -ssim_loss(rep_8_reco, rep_9_reco)
                    scores = score_0 + score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8 + score_9
                    loss = torch.mean(scores)
                    # Zero the network parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                elif self.ae_loss_type == 'mnist_HAE':

                    scores = torch.sum((x - x_reco) ** 2, dim=tuple(range(1, x_reco.dim())))+  \
                              torch.sum((rep_0 - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))) + \
                              torch.sum((rep_1 - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))) + \
                              torch.sum((rep_2 - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))) + \
                              torch.sum((rep_3 - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))) + \
                              torch.sum((rep_4) ** 2, dim=tuple(range(1, rep_4.dim())))
                elif self.ae_loss_type == 'mnist_HAE_ssim':
                    scores = -ssim_loss(x,x_reco) \
                             -ssim_loss(rep_0, rep_0_reco) \
                             -ssim_loss(rep_1, rep_1_reco) \
                             -ssim_loss(rep_2, rep_2_reco) \
                             -ssim_loss(rep_3, rep_3_reco)
                elif self.ae_loss_type == 'ssim':
                    scores = -ssim_loss(inputs, outputs)
                    loss = torch.mean(scores)
                    # Zero the network parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                    loss = torch.mean(scores)
                    # Zero the network parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)
        net = net.to(self.device)

        # Get test data loader
        train_loader, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average= False)
        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()


        with torch.no_grad():
            errors = []
            for data in train_loader:
                if self.dataset_name == 'object' or self.dataset_name == 'texture':
                    inputs, labels, idx = data
                else:
                    inputs, labels, idx = data

                inputs = inputs.to(self.device)

                if self.ae_loss_type == 'object_HAE' or self.ae_loss_type == 'object_HAE_ssim':
                    x_reco, rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, \
                    rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = ae_net(inputs)
                elif self.ae_loss_type == 'texture_HAE' or self.ae_loss_type == 'texture_HAE_ssim':
                    rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, \
                    rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = ae_net(inputs)
                    batchsize, chanle, l, h = inputs.size()
                    rep_0_list =[]; rep_0_reco_list =[]; rep_1_list =[] ;rep_1_reco_list =[]; rep_2_list =[]; rep_2_reco_list =[]; rep_3_list =[]; rep_3_reco_list =[]; rep_4_list =[]; rep_4_reco_list =[]; \
                    rep_5_list =[]; rep_5_reco_list =[]; rep_6_list =[]; rep_6_reco_list =[]; rep_7_list =[]; rep_7_reco_list =[]; rep_8_list =[]; rep_8_reco_list =[]; rep_9_list =[]; rep_9_reco_list =[];rep_10_list =[]
                    for i in range(0,l,128):
                        for j in range(0,h,128):
                            rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, \
                            rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = ae_net(inputs)
                            rep_0_list.append(rep_0_reco)
                            rep_1_list.append(rep_1_reco)
                            rep_2_list.append(rep_2_reco)
                            rep_3_list.append(rep_3_reco)
                            rep_4_list.append(rep_4_reco)
                            rep_5_list.append(rep_5_reco)
                            rep_6_list.append(rep_6_reco)
                            rep_7_list.append(rep_7_reco)
                            rep_8_list.append(rep_8_reco)
                            rep_9_list.append(rep_9_reco)

                    rep_0 = torch.cat(rep_0_list, 0)
                    rep_1 = torch.cat(rep_1_list, 0)
                    rep_2 = torch.cat(rep_2_list, 0)
                    rep_3 = torch.cat(rep_3_list, 0)
                    rep_4 = torch.cat(rep_4_list, 0)
                    rep_5 = torch.cat(rep_5_list, 0)
                    rep_6 = torch.cat(rep_6_list, 0)
                    rep_7 = torch.cat(rep_7_list, 0)
                    rep_8 = torch.cat(rep_8_list, 0)
                    rep_9 = torch.cat(rep_9_list, 0)

                elif self.ae_loss_type == 'mnist_HAE' or self.ae_loss_type == 'mnist_HAE_ssim':
                    x, x_reco, rep_0, rep_0_reco, rep_1, rep_1_reco, rep_2, rep_2_reco, rep_3, rep_3_reco, rep_4= ae_net(inputs)
                else:
                    outputs = ae_net(inputs)

                if self.ae_loss_type == 'object_HAE':
                    temp = []
                    temp.append(torch.sum((inputs - x_reco) ** 2, dim=tuple(range(1, x_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_4_reco) ** 2, dim=tuple(range(1, rep_4_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_5_reco) ** 2, dim=tuple(range(1, rep_5_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_6_reco) ** 2, dim=tuple(range(1, rep_6_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_7_reco) ** 2, dim=tuple(range(1, rep_7_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_8_reco) ** 2, dim=tuple(range(1, rep_8_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_9_reco) ** 2, dim=tuple(range(1, rep_9_reco.dim()))).cpu().numpy())
                    temp = np.array(temp)
                    for i in range(len(temp[0])):
                        errors.append(temp[:,i])
                elif self.ae_loss_type == 'object_HAE_ssim':
                    temp = []
                    temp.append(-ssim_loss(inputs, x_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_0_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_1_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_2_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_3_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_4_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_5_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_6_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_7_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_8_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_9_reco).cpu().numpy())
                    temp = np.array(temp)
                    for i in range(len(temp[0])):
                        errors.append(temp[:,i])
                elif self.ae_loss_type == 'texture_HAE':

                    temp = []
                    temp.append(torch.sum((inputs - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_4_reco) ** 2, dim=tuple(range(1, rep_4_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_5_reco) ** 2, dim=tuple(range(1, rep_5_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_6_reco) ** 2, dim=tuple(range(1, rep_6_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_7_reco) ** 2, dim=tuple(range(1, rep_7_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_8_reco) ** 2, dim=tuple(range(1, rep_8_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((inputs - rep_9_reco) ** 2, dim=tuple(range(1, rep_9_reco.dim()))).cpu().numpy())
                    temp = np.array(temp)
                    for i in range(len(temp[0])):
                        errors.append(temp[:,i])
                elif self.ae_loss_type == 'texture_HAE_ssim':
                    temp = []
                    temp.append(-ssim_loss(inputs, rep_0_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_1_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_2_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_3_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_4_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_5_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_6_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_7_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_8_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_9_reco).cpu().numpy())
                    temp = np.array(temp)
                    for i in range(len(temp[0])):
                        errors.append(temp[:,i])
                elif self.ae_loss_type == 'mnist_HAE':

                    temp = []
                    temp.append(torch.sum((x - x_reco) ** 2, dim=tuple(range(1, x_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((rep_0 - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((rep_1 - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((rep_2 - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((rep_3 - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((rep_4) ** 2, dim=tuple(range(1, rep_4.dim()))).cpu().numpy())
                    temp = np.array(temp)
                    for i in range(len(temp[0])):
                        errors.append(temp[:,i])
                elif self.ae_loss_type == 'mnist_HAE_ssim':
                    temp = []
                    temp.append(-ssim_loss(x, x_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_0, rep_0_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_1, rep_1_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_2, rep_2_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_3, rep_3_reco).cpu().numpy())
                    temp = np.array(temp)
                    for i in range(len(temp[0])):
                        errors.append(temp[:,i])

            errors = np.array(errors)
            # mean = np.mean(errors,0)
            # std = np.std(errors,0)
            # errors = (errors-mean)/std
            tree = KDTree(errors, leaf_size=40)  # doctest: +SKIP

            total_inputs = []
            if self.dataset_name == 'object':
                total_rec = [[],[],[],[],[],[],[],[],[],[],[]]
            elif self.dataset_name == 'texture':
                total_rec = [[], [], [], [], [], [], [], [], [], []]

            for data in test_loader:
                if self.dataset_name == 'object' or self.dataset_name =='texture':
                    inputs, labels, idx = data
                else:
                    inputs, labels, idx = data
                inputs = inputs.to(self.device)

                # np.save('./log/object/'+str(dataset.normal_classes)+'/Images.npy', inputs.cpu().numpy())
                # exit(0)

                if self.ae_loss_type == 'object_HAE' or self.ae_loss_type == 'object_HAE_ssim':
                    for i in inputs.cpu().numpy():
                        total_inputs.append(i)
                    x_reco, rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, \
                    rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = ae_net(inputs)
                    for i in x_reco.cpu().numpy():
                        total_rec[0].append(i)
                    for i in rep_0_reco.cpu().numpy():
                        total_rec[1].append(i)
                    for i in rep_1_reco.cpu().numpy():
                        total_rec[2].append(i)
                    for i in rep_2_reco.cpu().numpy():
                        total_rec[3].append(i)
                    for i in rep_3_reco.cpu().numpy():
                        total_rec[4].append(i)
                    for i in rep_4_reco.cpu().numpy():
                        total_rec[5].append(i)
                    for i in rep_5_reco.cpu().numpy():
                        total_rec[6].append(i)
                    for i in rep_6_reco.cpu().numpy():
                        total_rec[7].append(i)
                    for i in rep_7_reco.cpu().numpy():
                        total_rec[8].append(i)
                    for i in rep_8_reco.cpu().numpy():
                        total_rec[9].append(i)
                    for i in rep_9_reco.cpu().numpy():
                        total_rec[10].append(i)
                    continue
                elif self.ae_loss_type == 'texture_HAE' or self.ae_loss_type == 'texture_HAE_ssim':
                    batchsize, chanle, l, h = inputs.size()

                    rep_0_list = [];rep_1_list = [];rep_2_list = [];rep_3_list = [];rep_4_list = [];rep_5_list = [];rep_6_list = [];rep_7_list = [];rep_8_list = [];rep_9_list = [];input_x_list = [];

                    for i in range(0, l, 128):
                        for j in range(0, h, 128):
                            input_x = inputs[:,:,i:i+128,j:j+128]
                            rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = ae_net(input_x)
                            rep_0_list.append(rep_0_reco)
                            rep_1_list.append(rep_1_reco)
                            rep_2_list.append(rep_2_reco)
                            rep_3_list.append(rep_3_reco)
                            rep_4_list.append(rep_4_reco)
                            rep_5_list.append(rep_5_reco)
                            rep_6_list.append(rep_6_reco)
                            rep_7_list.append(rep_7_reco)
                            rep_8_list.append(rep_8_reco)
                            rep_9_list.append(rep_9_reco)
                            input_x_list.append(input_x)

                    rep_0 = torch.cat(rep_0_list, 0);rep_1 = torch.cat(rep_1_list, 0);rep_2 = torch.cat(rep_2_list, 0);rep_3 = torch.cat(rep_3_list, 0);rep_4 = torch.cat(rep_4_list, 0);rep_5 = torch.cat(rep_5_list, 0);rep_6 = torch.cat(rep_6_list, 0);rep_7 = torch.cat(rep_7_list, 0);rep_8 = torch.cat(rep_8_list, 0);rep_9 = torch.cat(rep_9_list, 0);input_x_list = torch.cat(input_x_list,0)

                elif self.ae_loss_type == 'mnist_HAE' or self.ae_loss_type == 'mnist_HAE_ssim':
                    x, x_reco, rep_0, rep_0_reco, rep_1, rep_1_reco, rep_2, rep_2_reco, rep_3, rep_3_reco, rep_4= ae_net(inputs)

                else:
                    outputs = ae_net(inputs)

                if self.ae_loss_type == 'object_HAE':
                    temp = []
                    temp.append(self.compute_local_error(inputs, x_reco, window_length= 16))
                    temp.append(self.compute_local_error(inputs, rep_0_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_1_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_2_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_3_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_4_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_5_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_6_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_7_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_8_reco, window_length=16))
                    temp.append(self.compute_local_error(inputs, rep_9_reco, window_length=16))
                    # temp.append(torch.sum((inputs - x_reco) ** 2, dim=tuple(range(1, x_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_4_reco) ** 2, dim=tuple(range(1, rep_4_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_5_reco) ** 2, dim=tuple(range(1, rep_5_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_6_reco) ** 2, dim=tuple(range(1, rep_6_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_7_reco) ** 2, dim=tuple(range(1, rep_7_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_8_reco) ** 2, dim=tuple(range(1, rep_8_reco.dim()))).cpu().numpy())
                    # temp.append(
                    #     torch.sum((inputs - rep_9_reco) ** 2, dim=tuple(range(1, rep_9_reco.dim()))).cpu().numpy())

                    # temp = np.array(temp)
                    # test_x = []
                    # for i in range(len(temp[0])):
                    #     test_x.append(temp[:,i])
                    test_x = torch.cat(temp, 1)
                    test_x = test_x.cpu().numpy()
                    scores = np.sum(test_x, 1)
                    dist, ind = tree.query(test_x, k=2)
                    avg_dist = np.mean(dist,1)
                elif self.ae_loss_type == 'object_HAE_ssim':
                    scores = -ssim_loss(x,x_reco) \
                             -ssim_loss(rep_0, rep_0_reco) \
                             -ssim_loss(rep_1, rep_1_reco) \
                             -ssim_loss(rep_2, rep_2_reco) \
                             -ssim_loss(rep_3, rep_3_reco)\
                             -ssim_loss(rep_4, rep_4_reco) \
                             -ssim_loss(rep_5, rep_5_reco) \
                             -ssim_loss(rep_6, rep_6_reco) \
                             -ssim_loss(rep_7, rep_7_reco) \
                             -ssim_loss(rep_8, rep_8_reco)
                    temp = []
                    temp.append(-ssim_loss(inputs, x_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_0_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_1_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_2_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_3_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_4_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_5_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_6_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_7_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_8_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_9_reco).cpu().numpy())
                    temp = np.array(temp)
                    test_x = []
                    for i in range(len(temp[0])):
                        test_x.append(temp[:, i])
                    test_x = np.array(test_x)
                    #test_x = torch.cat(temp,dim=1)
                    test_x = np.array(test_x)
                    scores = np.sum(test_x, 1)
                    dist, ind = tree.query(test_x, k=10)
                    avg_dist = np.mean(dist,1)
                elif self.ae_loss_type == 'texture_HAE':
                    for i in input_x_list.cpu().numpy():
                        total_inputs.append(i)
                    for i in rep_0.cpu().numpy():
                        total_rec[0].append(i)
                    for i in rep_1.cpu().numpy():
                        total_rec[1].append(i)
                    for i in rep_2.cpu().numpy():
                        total_rec[2].append(i)
                    for i in rep_3.cpu().numpy():
                        total_rec[3].append(i)
                    for i in rep_4.cpu().numpy():
                        total_rec[4].append(i)
                    for i in rep_5.cpu().numpy():
                        total_rec[5].append(i)
                    for i in rep_6.cpu().numpy():
                        total_rec[6].append(i)
                    for i in rep_7.cpu().numpy():
                        total_rec[7].append(i)
                    for i in rep_8.cpu().numpy():
                        total_rec[8].append(i)
                    for i in rep_9.cpu().numpy():
                        total_rec[9].append(i)
                    temp = []
                    temp.append(torch.sum((input_x_list - rep_0) ** 2, dim=tuple(range(1, rep_0.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_1) ** 2, dim=tuple(range(1, rep_1.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_2) ** 2, dim=tuple(range(1, rep_2.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_3) ** 2, dim=tuple(range(1, rep_3.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_4) ** 2, dim=tuple(range(1, rep_4.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_5) ** 2, dim=tuple(range(1, rep_5.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_6) ** 2, dim=tuple(range(1, rep_6.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_7) ** 2, dim=tuple(range(1, rep_7.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_8) ** 2, dim=tuple(range(1, rep_8.dim()))).cpu().numpy())
                    temp.append(torch.sum((input_x_list - rep_9) ** 2, dim=tuple(range(1, rep_9.dim()))).cpu().numpy())
                    temp = np.array(temp)
                    test_x = []
                    for i in range(len(temp[0])):
                        test_x.append(temp[:,i])
                    test_x = np.array(test_x)
                    scores = np.sum(test_x, 1)
                    dist, ind = tree.query(test_x, k=10)
                    avg_dist = np.mean(dist,1)
                elif self.ae_loss_type == 'texture_HAE_ssim':
                    scores = -ssim_loss(rep_0, rep_0_reco) \
                             -ssim_loss(rep_1, rep_1_reco) \
                             -ssim_loss(rep_2, rep_2_reco) \
                             -ssim_loss(rep_3, rep_3_reco)\
                             -ssim_loss(rep_4, rep_4_reco) \
                             -ssim_loss(rep_5, rep_5_reco) \
                             -ssim_loss(rep_6, rep_6_reco) \
                             -ssim_loss(rep_7, rep_7_reco) \
                             -ssim_loss(rep_8, rep_8_reco)
                    temp = []
                    temp.append(-ssim_loss(inputs, rep_0_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_1_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_2_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_3_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_4_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_5_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_6_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_7_reco).cpu().numpy())
                    temp.append(-ssim_loss(inputs, rep_8_reco).cpu().numpy())
                    temp = np.array(temp)
                    test_x = []
                    for i in range(len(temp[0])):
                        test_x.append(temp[:, i])
                    test_x = np.array(test_x)
                    scores = np.sum(test_x, 1)
                    dist, ind = tree.query(test_x, k=10)
                    avg_dist = np.mean(dist,1)
                elif self.ae_loss_type == 'mnist_HAE':
                    scores =  torch.sum((x - x_reco) ** 2, dim=tuple(range(1, x_reco.dim())))+  \
                              torch.sum((rep_0 - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))) + \
                              torch.sum((rep_1 - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))) + \
                              torch.sum((rep_2 - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))) + \
                              torch.sum((rep_3 - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))) + \
                              torch.sum((rep_4) ** 2, dim=tuple(range(1, rep_4.dim())))
                    temp = []
                    temp.append(torch.sum((x - x_reco) ** 2, dim=tuple(range(1, x_reco.dim()))).cpu().numpy())
                    temp.append(
                        torch.sum((rep_0 - rep_0_reco) ** 2, dim=tuple(range(1, rep_0_reco.dim()))).cpu().numpy())
                    temp.append(
                        torch.sum((rep_1 - rep_1_reco) ** 2, dim=tuple(range(1, rep_1_reco.dim()))).cpu().numpy())
                    temp.append(
                        torch.sum((rep_2 - rep_2_reco) ** 2, dim=tuple(range(1, rep_2_reco.dim()))).cpu().numpy())
                    temp.append(
                        torch.sum((rep_3 - rep_3_reco) ** 2, dim=tuple(range(1, rep_3_reco.dim()))).cpu().numpy())
                    temp.append(torch.sum((rep_4) ** 2, dim=tuple(range(1, rep_4.dim()))).cpu().numpy())
                    temp = np.array(temp)
                    test_x = []
                    for i in range(len(temp[0])):
                        test_x.append(temp[:,i])
                    test_x = np.array(test_x)
                    scores = np.sum(test_x,1)
                    dist, ind = tree.query(test_x, k=10)
                    avg_dist = np.mean(dist,1)
                elif self.ae_loss_type == 'mnist_HAE_ssim':
                    scores = -ssim_loss(x,x_reco) \
                             -ssim_loss(rep_0, rep_0_reco) \
                             -ssim_loss(rep_1, rep_1_reco) \
                             -ssim_loss(rep_2, rep_2_reco) \
                             -ssim_loss(rep_3, rep_3_reco)
                    temp = []
                    temp.append(-ssim_loss(x, x_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_0, rep_0_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_1, rep_1_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_2, rep_2_reco).cpu().numpy())
                    temp.append(-ssim_loss(rep_3, rep_3_reco).cpu().numpy())
                    temp = np.array(temp)
                    test_x = []
                    for i in range(len(temp[0])):
                        test_x.append(temp[:, i])
                    test_x = np.array(test_x)
                    scores = np.sum(test_x, 1)
                    dist, ind = tree.query(test_x, k=10)
                    avg_dist = np.mean(dist,1)
                elif self.ae_loss_type == 'ssim':
                    scores = -ssim_loss(inputs,outputs)
                    scores = scores.cpu().numpy()
                else:
                    scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                    scores = scores.cpu().numpy()
                loss = np.mean(scores)

                # Save triple of (idx, label, score) in a list
                if self.dataset_name == 'object' or self.dataset_name == 'texture':
                    if self.ae_loss_type == 'object_HAE_ssim' or self.ae_loss_type == 'object_HAE'\
                            or self.ae_loss_type == 'texture_HAE_ssim' or self.ae_loss_type == 'texture_HAE':
                        idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                    labels.cpu().data.numpy().tolist(),
                                                    scores.tolist(),
                                                    avg_dist.tolist(),
                                                    test_x.tolist()))

                    else:
                        idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.tolist(),
                                                avg_dist.tolist()))
                else:
                    if self.ae_loss_type == 'mnist_HAE_ssim' or self.ae_loss_type == 'mnist_HAE':
                        idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                    labels.cpu().data.numpy().tolist(),
                                                    scores.cpu().data.numpy().tolist(),
                                                    avg_dist.tolist(),
                                                    test_x.tolist()))
                    else:
                        idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1
            print(np.shape(total_inputs))
            print(np.shape(total_rec))
            np.save('./log/'+self.dataset_name+'/' + str(dataset.normal_classes) + '/Images.npy', np.array(total_inputs))
            for i in range(len(total_rec)):
                np.save('./log/'+self.dataset_name+'/' + str(dataset.normal_classes) + '/Data_Reconstruction_'+str(i)+'.npy', np.array(total_rec[i]))
            exit()
        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))
        if self.dataset_name == 'object' or self.dataset_name == 'texture':
            if self.ae_loss_type == 'object_HAE_ssim' or self.ae_loss_type == 'object_HAE' or self.ae_loss_type == 'texture_HAE_ssim' or self.ae_loss_type == 'texture_HAE':
                _, labels, scores, avg_dist, test_x = zip(*idx_label_score)
            else:
                _, labels, scores,avg_dist = zip(*idx_label_score)
        else:
            if self.ae_loss_type == 'mnist_HAE_ssim' or self.ae_loss_type == 'mnist_HAE' or self.ae_loss_type == 'mnist_HAE_ssim' or self.ae_loss_type == 'minist_object_HAE':
                _, labels, scores, avg_dist, test_x = zip(*idx_label_score)
            else:
                _, labels, scores = zip(*idx_label_score)


        labels = np.array(labels)
        scores = np.array(scores)
        avg_dist = np.array(avg_dist)
        test_x = np.array(test_x)

        labels[labels>0]=1
        if self.ae_loss_type == 'object_HAE_ssim' or self.ae_loss_type == 'object_HAE'or self.ae_loss_type == 'texture_HAE_ssim' or self.ae_loss_type == 'texture_HAE' or self.ae_loss_type == 'mnist_HAE_ssim' or self.ae_loss_type == 'mnist_HAE':
            auc = roc_auc_score(labels, scores)
            logger.info('HAE AUC: {:.2f}%'.format(100. * auc))

            auc = roc_auc_score(labels, avg_dist)
            logger.info('Test KNN set AUC: {:.2f}%'.format(100. * auc))
            _, num_feature = np.shape(test_x)
            for i in range(num_feature):
                auc = roc_auc_score(labels, test_x[:,i])
                logger.info('Test '+str(i)+'th set AUC: {:.2f}%'.format(100. * auc))
        else:
            auc = roc_auc_score(labels, scores)
            logger.info('Test set AUC: {:.2f}%'.format(100. * auc))



        test_time = time.time() - start_time
        self.test_scores = idx_label_score
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
        plot_reconstruction(net, self.device, self.ae_loss_type, dataset.normal_classes)

    def compute_local_error(self, input, reconstruction , window_length, stride = 4 ):
        # compute the maximum of local error
        b, c, l, h = input.size()
        score_max = 0
        error_list = []
        for i in range(0,l-window_length, stride):
            for j in range(0,h-window_length, stride):
                score_temp = torch.sum((input[:,:,i:i+window_length, j:j+window_length]-reconstruction[:,:,i:i+window_length, j:j+window_length])** 2, dim=tuple(range(1, input.dim())))
                score_temp = score_temp.unsqueeze(1)
                error_list.append(score_temp)
        score_tensor = torch.cat(error_list, dim=1)
        ah, bh =torch.max(score_tensor, 1)

        return  bh.unsqueeze(1)
