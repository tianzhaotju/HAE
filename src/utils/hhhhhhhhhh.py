import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_ssim import SSIM
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from utils.visualization.plot_images_grid import plot_images_grid
from sklearn.neighbors import KernelDensity, KDTree

#############################
#  object  texture
cat = 'texture'
normal_classes = '0'
sub_cat = 'broken_large'
# ssim  l2
loss_type = 'ssim'
#############################
Mvtec_list = ['carpet', 'grid', 'leather', 'tile', 'wood',
              'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
              'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
good_num =  {'0':28, '1':21,'2':32, '3':33, '4':19, '5': 20, '6':58, '7':23,'8':40 , '9':22,  '10':26,  '11':41, '12':12, '13':60, '14':32}
label_num = {'0':117,"1":78,'2':124,'3':117,'4':79, '5': 40, '6':150,'7':46,'8':110, '9':92, '10':167, '11':160, '12':42, '13':100, '14':143}
reverse_flag = {'0':0, "1":0,'2':0,'3':0,'4':0, '5': 1, '6':0,'7':1,'8':0, '9':0, '10':0, '11':1, '12':0, '13':0, '14':0}
# label_num = {'8':110,'7':132,'6':150,"1":78,'2':124,'3':117,'4':79,'0':117}

def plot_pics():
    with torch.no_grad():

        data_native_6 = np.load('../.././log/object/6/Images.npy')
        data_native_7 = np.load('../.././log/object/7/Images.npy')
        data_native_8 = np.load('../.././log/object/8/Images.npy')
        data_native_9 = np.load('../.././log/object/9/Images.npy')
        data_native_10 = np.load('../.././log/object/10/Images.npy')

        good = []
        for i in range(5):
            temp = np.array(data_native_6[i])
            good.append(temp)
        for i in range(5):
            temp = np.array(data_native_7[i])
            good.append(temp)
        for i in range(5):
            temp = np.array(data_native_8[i])
            good.append(temp)
        for i in range(5):
            temp = np.array(data_native_9[i])
            good.append(temp)
        for i in range(5):
            temp = np.array(data_native_10[i])
            good.append(temp)
        good = np.array(good)
        good = torch.from_numpy(good)
        plot_images_grid(good, export_img='./pictures/good.eps', title='Normal', nrow=5, padding=5)

        abnormal = []
        for i in range(60,65):
            temp = np.array(data_native_6[i])
            abnormal.append(temp)
        for i in range(23,28):
            temp = np.array(data_native_7[i])
            abnormal.append(temp)
        for i in range(50,55):
            temp = np.array(data_native_8[i])
            abnormal.append(temp)
        for i in range(60,65):
            temp = np.array(data_native_9[i])
            abnormal.append(temp)
        for i in range(135,140):
            temp = np.array(data_native_10[i])
            abnormal.append(temp)
        abnormal = np.array(abnormal)
        abnormal = torch.from_numpy(abnormal)
        plot_images_grid(abnormal, export_img='./pictures/abnormal.eps', title='Abnormal', nrow=5, padding=5)

        # a = 28
        # for i in range(a*16,a*16+16):
        #     temp = np.array(data_native[i]).transpose([1, 2, 0])
        #     plt.imshow(temp)
        #     plt.title('naive')
        #     plt.show()
            # for j in range(np.shape(reconstructions)[0]):
            #     temp = np.array(reconstructions[j][i]).transpose([1, 2, 0])
            #     plt.imshow(temp)
            #     plt.title(str(i))
            #     plt.show()
            # break
        # exit()
        #     temp = np.array(reconstructions[0][i]).transpose([1, 2, 0])
        #     plt.imshow(temp)
        #     plt.title(str(i))
        #     plt.show()
        #     temp = np.array(data_native[i]).transpose([1, 2, 0])
        #     plt.imshow(temp)
        #     plt.title(str(i))
        #     plt.show()
        # exit()


def main(normal_classes):

    with torch.no_grad():
        ae_loss_type = 'l2'
        path = '../.././log/'+cat+'/' + normal_classes
        data_native_train = np.load(path+'/Images_train.npy')
        data_native = np.load(path + '/Images.npy')
        reconstructions_train = []
        reconstructions = []
        if cat == 'object':
            for i in range(11):
                reconstructions_train.append(np.load('../.././log/' + cat + '/' + normal_classes + '/Data_Reconstruction_' + str(i) +'_train'+ '.npy'))
                reconstructions.append(np.load('../.././log/' + cat + '/' + normal_classes + '/Data_Reconstruction_' + str(i) + '.npy'))
        elif cat == 'texture':
            for i in range(10):
                reconstructions_train.append(np.load('../.././log/' + cat + '/' + normal_classes + '/Data_Reconstruction_' + str(i) +'_train'+ '.npy'))
                reconstructions.append(np.load('../.././log/' + cat + '/' + normal_classes + '/Data_Reconstruction_' + str(i) + '.npy'))

        features_train_s = []
        features_train_cal = []
        features_s = []
        features_cal = []
        if ae_loss_type == 'ssim':
            pass
        elif ae_loss_type == 'l2':
            for i in range(len(reconstructions_train)):
                if i == 0:
                    features_train_s.append(abs(data_native_train - reconstructions_train[i]))
                else:
                    features_train_s.append(abs(reconstructions_train[i - 1] - reconstructions_train[i]))
                features_train_cal.append(abs(data_native_train - reconstructions_train[i]))

            for i in range(len(reconstructions)):
                if i == 0:
                    features_s.append(abs(data_native - reconstructions[i]))
                else:
                    features_s.append(abs(reconstructions[i-1]-reconstructions[i]))
                features_cal.append(abs(data_native-reconstructions[i]))

            pre_s = KNN_dist(features_train_s,features_s)
            pre_cal = KNN_dist(features_train_cal,features_cal)
            print(normal_classes)
            print('layer error KNN', pre_s)
            print('calculated error KNN', pre_cal)

def compute_local_error(error, windows_length = 20, stride = 4):
    l,h = np.shape(error)
    error_max = 0
    for i in range(0, l-windows_length, stride):
        for j in range(0, h-windows_length, stride):
            errror_mu = np.mean(error[ i:i+windows_length, j:j+windows_length])
            errror_std = np.std(error[i:i + windows_length, j:j + windows_length])
            temp_score = errror_mu*(errror_std**.25)
            if temp_score > error_max:
                error_max = temp_score
    return error_max

def KNN_dist(features_train, features):
    scores = []
    for j in range(np.shape(features_train)[1]):
        scores_temp = []
        for i in range(np.shape(features_train)[0]):
            m = features_train[i][j]
            m_single = np.sum(m, 0)
            if loss_type == 'ssim':
                score = compute_local_error(m_single)
            elif loss_type == 'l2':
                score = np.sum(m_single)
            scores_temp.append(score)
        scores.append(scores_temp)
    tree = KDTree(np.array(scores), leaf_size=40)  # doctest: +SKIP

    scores_test = []
    for j in range(np.shape(features)[1]):
        scores_temp = []
        for i in range(np.shape(features)[0]):
            m = features[i][j]
            m_single = np.sum(m, 0)
            if loss_type == 'ssim':
                score = compute_local_error(m_single)
            elif loss_type == 'l2':
                score = np.sum(m_single)
            scores_temp.append(score)
        scores_test.append(scores_temp)
    dist, ind = tree.query(np.array(scores_test), k=2)
    avg_dist = np.mean(dist, 1)
    return avg_dist


if __name__ == '__main__':
    for i in range(5,15):
        main(str(i))
    # plot_pics()



#texture 0-4
# python src/main.py texture  texture_hae ./log/texture ./data/Mvtec/ --objective deep-GMM --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --seed -1 --ae_lr 0.0001 --ae_lr_milestone 50 --ae_batch_size 32 --ae_weight_decay 0.5e-3 --ae_loss_type 'texture_HAE' --ae_only True --normal_class 1  --ae_n_epochs 20 --device cuda:0 --ae_test_only False
