from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from utils.pytorch_ssim import  pytorch_ssim
import logging
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from math import pi


# def to_var(x, volatile=False):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x, volatile=volatile)

# class Cholesky(torch.autograd.Function):
#     def forward(ctx, a):
#         l = torch.potrf(a, False)
#         ctx.save_for_backward(l)
#         return l
#     def backward(ctx, grad_output):
#         l, = ctx.saved_variables
#         linv = l.inverse()
#         inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
#             1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
#         s = torch.mm(linv.t(), torch.mm(inner, linv))
#         return s


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, dataset_name: str = 'mnist'):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader, dataset_name)

        assert objective in ('one-class', 'soft-boundary', 'deep-GMM', 'hybrid'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # GMM Parameters

        # Cifar_ELU

        self.energy_lambda = 0.1
        self.cov_diag_lambda = 0.00

        self.ssim_lambda = 10

        self.l2_lambda = 1



        self.l2_lambda_test = 0.01


        # Cifar
        # self.n_components = 4
        # self.n_features = 16
        # self.cov_diag_lambda = 0.000005

        # Mnist
        # self.n_components = 10
        # self.n_features = 32
        # self.cov_diag_lambda = 0.000005

        self.eps = 1.e-6
        self.gamma_sum = 0
        self.iteration = 0
        self.isTesting = False
        self.pretrain_epoch = 100


        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def compute_gmm_params(self, z, gamma):

        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = self.to_var(self.phi)
        if mu is None:
            mu = self.to_var(self.mu)
        if cov is None:
            cov = self.to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12

        #phi size: K
        #mu size:   K x D
        #conv size: K x D x D

        for i in range(k):
            # K x D x D
            #??
            #cov_k = cov[i] + self.to_var(torch.eye(D)*eps)
            cov_k = cov[i]
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            # K
            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()*(2*np.pi)))
            #det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K

        det_cov = self.to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        #det_cov = torch.cat(det_cov).cuda()

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)

        exp_term_save = exp_term_tmp
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        #sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)


        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov) +eps).unsqueeze(0), dim = 1))

        ##???
        sample_energy = - torch.log(torch.sum(phi.unsqueeze(0) * torch.exp(exp_term_save) / (torch.sqrt(det_cov) +eps).unsqueeze(0), dim = 1))
        ##???

        #sample_energy  = -max_val.squeeze() - torch.log(torch.sum((phi.unsqueeze(0) * exp_term + eps*1.e-20)/ (torch.sqrt((2*np.pi)**D * det_cov)+ eps).unsqueeze(0), dim=1) )
        #

        # for j in range(10):
        #     if sample_energy[j].data <0:
        #         print('-------------------------------------------------------')
        #         print('----------------------')
        #         for i in range(len(torch.exp(exp_term_save)[j].data)):
        #             print('Energy', sample_energy[j].data, 'Exp term: ', torch.exp(exp_term_save)[j][i].data, 'Det term: ', torch.sqrt(D *det_cov)[i].data)
        #         print('----------------------')
        #         print('-------------------------------------------------------')

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self, dataset: BaseADDataset, net: BaseNet):

        # #?????
        # for k,v in net.named_parameters():
        #     if k!= 'dense1.weight' and k!= 'dense1.bias':
        #         v.requires_grad = False
        # #?????

        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        #optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,amsgrad=self.optimizer_name == 'amsgrad')
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr, weight_decay=self.weight_decay,amsgrad=self.optimizer_name == 'amsgrad')


        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        for epoch in range(self.n_epochs):


            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            loss_rec = 0.0
            loss_epoch = 0.0
            loss_diag = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                if self.dataset_name == 'object' or self.dataset_name == 'texture':
                    inputs, _, _ = data
                else:
                    inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()
                sample_energy = 0
                cov_diag = 0
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, category, resconstruction = net(inputs)
                if self.objective == 'deep-GMM':

                    phi, mu, cov = self.compute_gmm_params(outputs, category)

                    sample_energy, cov_diag = self.compute_energy(outputs, phi=phi, mu=mu, cov=cov, size_average=True)
                    if self.ae_loss_type == 'ssim':
                        rescon_error = -ssim_loss(inputs,resconstruction)
                        rescon_error = self.ssim_lambda*rescon_error
                    else:
                        rescon_error = torch.sum((resconstruction - inputs) ** 2, dim=tuple(range(1, resconstruction.dim())))
                        rescon_error = self.l2_lambda* rescon_error
                    rescon_loss = torch.mean(rescon_error)

                    loss = sample_energy+rescon_loss+self.cov_diag_lambda*cov_diag


                elif self.objective == 'soft-boundary':
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                elif self.objective == 'hybrid':
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.ae_loss_type == 'ssim':
                        rescon_error = -ssim_loss(inputs,resconstruction)
                        rescon_error = self.ssim_lambda*rescon_error
                    else:
                        rescon_error = torch.sum((resconstruction - inputs) ** 2, dim=tuple(range(1, resconstruction.dim())))
                        rescon_error = self.l2_lambda* rescon_error
                    rescon_loss = torch.mean(rescon_error)
                    dist_ave = torch.mean(dist)
                    dist = rescon_error+dist
                    loss = torch.mean(dist)

                else:
                    if self.ae_loss_type == 'ssim':
                        rescon_error = -ssim_loss(inputs,resconstruction)
                        rescon_error = self.ssim_lambda*rescon_error
                    else:
                        rescon_error = torch.sum((resconstruction - inputs) ** 2, dim=tuple(range(1, resconstruction.dim())))
                        rescon_error = self.l2_lambda* rescon_error
                    rescon_loss = torch.mean(rescon_error)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    dist_ave = torch.mean(dist)
                    loss = torch.mean(dist)


                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                if self.objective == 'deep-GMM':
                    loss_epoch += sample_energy.item()
                else:
                    loss_epoch += dist_ave.item()
                #loss_diag += self.cov_diag_lambda*cov_diag.item()
                loss_rec += rescon_loss.item()


                n_batches += 1
            scheduler.step()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            #logger.info('  Epoch {}/{}\t Time: {:.3f}\t Energy: {:.8f}, Cov_diag:  {:.8f},  Reconstrcion:  {:.8f}'.format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, loss_diag, loss_rec/ n_batches))
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Energy: {:.8f}, Reconstrcion:  {:.8f}'.format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches , loss_rec/ n_batches))
            self.l2_lambda_test = loss_epoch/loss_rec
        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):

        self.n_components = net.cate_dense_2
        self.n_features = net.rep_dim
        self.mu_test = torch.tensor(np.float32(np.zeros([1, self.n_components, self.n_features])), device=self.device)
        self.cov_test = torch.tensor(np.float32(np.zeros([1, self.n_components, self.n_features, self.n_features])),
                                     device=self.device)

        self.isTesting = True

        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        train_loader, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average= False)
        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                if self.dataset_name == 'object' or self.dataset_name =='texture':
                    inputs, labels, idx = data
                else:
                    inputs, labels, idx = data

                inputs = inputs.to(self.device)
                outputs, category, resconstruction = net(inputs)

                if self.objective == 'deep-GMM':
                    phi, mu, cov = self.compute_gmm_params(outputs, category)

                    batch_gamma_sum = torch.sum(category, dim=0)

                    self.gamma_sum += batch_gamma_sum

                    self.mu_test += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
                    self.cov_test += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

                    self.iteration += inputs.size(0)

                    train_phi = self.gamma_sum / self.iteration
                    train_mu = self.mu_test / self.gamma_sum.unsqueeze(-1)
                    train_cov = self.cov_test / self.gamma_sum.unsqueeze(-1).unsqueeze(-1)


                    train_cov = train_cov.squeeze(0)
                    train_mu = train_mu.squeeze(0)

            for data in test_loader:
                if self.dataset_name == 'object' or self.dataset_name =='texture':
                    inputs, labels, idx = data
                else:
                    inputs, labels, idx = data

                inputs = inputs.to(self.device)
                outputs, category, resconstruction = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                if self.objective == 'deep-GMM':
                    phi, _, _ = self.compute_gmm_params(outputs, category)
                    sample_energy, cov_diag = self.compute_energy(outputs, phi=phi, mu=train_mu, cov=train_cov,
                                                                  size_average=False)

                    if self.ae_loss_type=='ssim':
                        rescon_error = -ssim_loss(inputs,resconstruction)
                        rescon_error=rescon_error*self.ssim_lambda
                    else:
                        rescon_error = torch.sum((resconstruction - inputs) ** 2, dim=tuple(range(1, resconstruction.dim())))
                        rescon_error = rescon_error*self.l2_lambda

                    scores = sample_energy+rescon_error

                    #scores = rescon_error
                elif self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                elif self.objective == 'hybrid':
                    if self.ae_loss_type == 'ssim':
                        rescon_error = -ssim_loss(inputs, resconstruction)
                        rescon_error = rescon_error * self.ssim_lambda
                    else:
                        rescon_error = torch.sum((resconstruction - inputs) ** 2,dim=tuple(range(1, resconstruction.dim())))
                        rescon_error = rescon_error * self.l2_lambda_test
                    sample_energy = dist
                    scores = dist + rescon_error


                else:
                    if self.ae_loss_type == 'ssim':
                        rescon_error = -ssim_loss(inputs, resconstruction)
                        rescon_error = rescon_error * self.ssim_lambda
                    else:
                        rescon_error = torch.sum((resconstruction - inputs) ** 2,
                                                 dim=tuple(range(1, resconstruction.dim())))
                        rescon_error = rescon_error * self.l2_lambda*1.5
                    sample_energy = dist
                    scores = dist

                # Save triples of (idx, label, score) in a list
                if self.dataset_name == 'object' or self.dataset_name == 'texture':
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist(),
                                                sample_energy.cpu().data.numpy().tolist(),
                                                rescon_error.cpu().data.numpy().tolist()))

                else:
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist(),
                                                sample_energy.cpu().data.numpy().tolist(),
                                                rescon_error.cpu().data.numpy().tolist()))


        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        if self.dataset_name == 'object' or self.dataset_name == 'texture':
            _, labels, scores, energy, rescon_error = zip(*idx_label_score)
        else:
              _, labels, scores, energy, rescon_error= zip(*idx_label_score)
        labels = np.array(labels)
        labels[labels>0] =1
        scores = np.array(scores)
        energy = np.array(energy)
        rescon_error = np.array(rescon_error)

        # for i in range(3,40):
        #     if labels[i] == 0:
        #         print('--------------------------------------')
        #         print('           ')
        #     print('labels:', labels[i], 'scores: ', scores[i],'energy: ',self.energy_lambda*energy[i], 'reconstruction: ', rescon_error[i])
        #
        #     if labels[i] == 0:
        #         print('           ')
        #         print('--------------------------------------')
        #
        # for i in range(len(scores)):
        #     if np.isnan(scores[i]) or np.isinf(scores[i]):
        #         scores[i] = 100
        #         print(labels[i])

        scores[scores>100] = 100


        self.test_auc = roc_auc_score(labels, rescon_error)
        logger.info('Test set reconstruction AUC: {:.2f}%'.format(100. * self.test_auc))

        self.test_auc = roc_auc_score(labels, energy)
        logger.info('Test set one-class AUC: {:.2f}%'.format(100. * self.test_auc))

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set hybrid AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                if self.dataset_name == 'object' or self.dataset_name == 'texture':
                    inputs, _ ,_= data
                else:
                    inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs, _, _ = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
