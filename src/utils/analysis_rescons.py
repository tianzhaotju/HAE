import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_ssim import SSIM
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#  object  texture
cat = 'texture'
normal_classes = '2'

Mvtec_list = ['carpet', 'grid', 'leather', 'tile', 'wood',
              'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
              'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
good_num = {'8':40,'6':58,'7':23,'1':21,'2':32,'3':33,'4':19,'0':28}
label_num = {'7':132,'6':150,"1":78,'2':124,'3':117,'4':79,'0':117}

def main():
    with torch.no_grad():
        ae_loss_type = 'l2'
        path = '../.././log/'+cat+'/' + normal_classes

        # np.save('./log/AnboT/Reconstruction_evaluation/native.npy', inputs.cpu().numpy())
        data_native = np.load(path+'/Images.npy')

        reconstructions = []
        if cat == 'object':
            for i in range(11):
                reconstruction_temp = np.load('../.././log/'+cat+'/' + normal_classes + '/Data_Reconstruction_'+str(i)+'.npy')
                reconstructions.append(reconstruction_temp)
        elif cat == 'texture':
            for i in range(10):
                reconstruction_temp = np.load('../.././log/'+cat+'/' + normal_classes + '/Data_Reconstruction_'+str(i)+'.npy')
                reconstructions.append(reconstruction_temp)

        # a = 33
        # for i in range(a*16,a*16+16):
        #     for j in range(np.shape(reconstructions)[0]):
        #         temp = np.array(reconstructions[j][i]).transpose([1, 2, 0])
        #         plt.imshow(temp)
        #         plt.title(str(i))
        #         plt.show()
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



        features = []
        if ae_loss_type == 'ssim':
            for reconstruction in reconstructions:
                pass
        else:
            for i in range(len(reconstructions)):
                reconstruction = reconstructions[i]
                # for i in range(524, len(reconstruction)):
                #     temp = np.array(reconstruction[i]).transpose([1,2,0])
                #     plt.imshow(temp)
                #     plt.title(str(i))
                #     plt.show()
                #     exit()
                #     break
                feature = (data_native - reconstruction)**2
                features.append(feature)
            # exit()

            # for feature in features:
            #     for i in range(698,len(feature)):
            #         temp = np.array(feature[i]).transpose([1,2,0])
            #         plt.imshow(temp)
            #         plt.title(str(i))
            #         plt.show()
            #         break


        if cat == 'object':
            for i in range(np.shape(features)[0]):
                scores = []
                for j in range(0,np.shape(features)[1]):
                    m = features[i][j]
                    m_single = np.sum(m, 0)
                    # score = compute_local_error(m_single)
                    score = np.sum(m_single)
                    scores.append(score)

                    # plt.imshow(m_single, cmap='gray')
                    # plt.title(str(score))
                    # plt.show()

                labels = np.ones([label_num[normal_classes]])
                labels[0:good_num[normal_classes]] = 0
                scores = np.array(scores)
                # print(scores)
                # print(labels)
                print(str(roc_auc_score(labels,scores)*100)+"%")

        elif cat == 'texture':
            total_score = np.zeros([label_num[normal_classes]])
            labels = np.ones([label_num[normal_classes]])
            labels[0:good_num[normal_classes]] = 0
            for i in range(np.shape(features)[0]):
                scores = []
                for j in range(0, np.shape(features)[1],16):
                    score = 0
                    for k in range(j,j+16):
                        m = features[i][k]
                        m_single = np.sum(m, 0)
                        # score += compute_local_error(m_single)
                        score += np.sum(m_single)
                        # score = max(np.sum(m_single),score)
                    scores.append(score)
                scores = np.array(scores)
                total_score+=scores
                # print(scores)
                # print(labels)
                # print(np.shape(scores))
                # print(np.shape(labels))
                # exit()
                # print(labels)
                # print(scores)
                print(str(roc_auc_score(labels, scores) * 100) + "%")
                # fpr,tpr,thresholds = roc_curve(labels, scores)
                # print(thresholds)
                # plt.plot(fpr,tpr)
                # plt.show()
            print("--------------------")
            # fpr, tpr, thresholds = roc_curve(labels, total_score)
            # print(thresholds)
            # plt.plot(fpr, tpr)
            # plt.show()
            print(str(roc_auc_score(labels, total_score) * 100) + "%")

        # [11,100,3,256,256]
        #for i in range(np.shape(features)[1]):
        #for i in range(np.shape(features)[1]):
        # for i in range(np.shape(features)[1]):
        #     # plt.imshow(np.transpose(data_native[i], (1, 2, 0)))
        #     # plt.savefig("../.././log/object/8/"+str(i)+'naive')
        #     for j in range(np.shape(features)[0]):
        #         m = features[j][i]
        #         m_single = np.sum(m, 0)
        #         # black = np.zeros([3,256,256])
        #         # m_single = (m_single-np.min(m_single))/(np.max(m_single)-np.min(m_single))
        #         # plt.imshow(m_single, cmap='gray')
        #         # m_top = np.sort(m_single)[::-1][0:40]
        #         # plt.title(str(np.sum(m_top))+' & '+str(compute_local_error(m_single)))
        #         score = compute_local_error(m_single)
        #         # plt.title(str(score))
        #         # plt.savefig("../.././log/object/8/" + str(i)+'_'+str(j)+'_rec')
        #         # f = open("../.././log/object/8/" + str(i)+'_'+str(j)+'_rec'+'.txt', 'w')
        #         # for line in m_single:
        #         #     for k in line:
        #         #         if float(k) >= 0.1:
        #         #             f.writelines(str(k)+'\n')

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

if __name__ == '__main__':
    main()

#texture 0-4
# python src/main.py texture  texture_hae ./log/texture ./data/Mvtec/ --objective deep-GMM --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --seed -1 --ae_lr 0.0001 --ae_lr_milestone 50 --ae_batch_size 32 --ae_weight_decay 0.5e-3 --ae_loss_type 'texture_HAE' --ae_only True --normal_class 1  --ae_n_epochs 20 --device cuda:0 --ae_test_only False
