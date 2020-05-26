import torch
import numpy as np
from utils.visualization.plot_images_grid import plot_images_grid
from utils.pytorch_ssim import  pytorch_ssim
import matplotlib.pyplot as plt

def plot_reconstruction(net,device,ae_loss_type, normal_classes):
    ssim_loss1 = pytorch_ssim.SSIM(window_size=11, size_average=False)
    with torch.no_grad():
        # np.save('./log/object/Reconstruction_evaluation/Images.npy', inputs.cpu().numpy()[-16:])
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



        path = './log/object/' + str(normal_classes)

        # np.save('./log/AnboT/Reconstruction_evaluation/native.npy', inputs.cpu().numpy())
        data_native = np.load(path+'/Images.npy')
        data_native = torch.from_numpy(data_native).to(device)

        reconstructions = []

        if ae_loss_type == 'object_HAE' or ae_loss_type == 'object_HAE_ssim':
            x_reco, rep_0_reco, rep_1_reco, rep_2_reco, rep_3_reco, rep_4_reco, \
            rep_5_reco, rep_6_reco, rep_7_reco, rep_8_reco, rep_9_reco = net(data_native)
            reconstructions.append(x_reco)
            reconstructions.append(rep_0_reco)
            reconstructions.append(rep_1_reco)
            reconstructions.append(rep_2_reco)
            reconstructions.append(rep_3_reco)
            reconstructions.append(rep_4_reco)
            reconstructions.append(rep_5_reco)
            reconstructions.append(rep_6_reco)
            reconstructions.append(rep_7_reco)
            reconstructions.append(rep_8_reco)
            reconstructions.append(rep_9_reco)


        elif ae_loss_type == 'texture_HAE' or ae_loss_type == 'texture_HAE_ssim':
            pass
        elif ae_loss_type == 'mnist_HAE' or ae_loss_type == 'mnist_HAE_ssim':
            pass
        else:
            reconstruction = net(data_native)
            reconstructions.append(reconstruction)

        # for i in range(3):
        #     data_native[:, i, :, :] *= std[normal_classes][i]
        #     data_native[:, i, :, :] += mean[normal_classes][i]
        #     data_native[:, i, :, :] *= 6
        #     data_native[:, i, :, :] += 3
        #     reconstruction[:, i, :, :] *= std[normal_classes][i]
        #     reconstruction[:, i, :, :] += mean[normal_classes][i]
        #     reconstruction[:, i, :, :] *= 6
        #     reconstruction[:, i, :, :] += 3

        features = []

        if ae_loss_type == 'ssim':
            for reconstruction in reconstructions:
                feature = -ssim_loss1(data_native, reconstruction)
                features.append(feature.cpu().numpy())
        else:
            for reconstruction in reconstructions:
                feature = (data_native - reconstruction) ** 2
                features.append(feature.cpu().numpy())

        plot_images_grid(data_native[0::4], export_img=path + '/Img_native', title='Images', nrow=4, padding=4)
        with torch.no_grad():
            np.save('./log/object/' + str(normal_classes) + '/Data_Images.npy', data_native.cpu().numpy())

        print(np.shape(features))

        features = np.array(features)
        data_native = data_native.cpu().numpy()

        # [11,100,3,256,256]
        # for i in range(np.shape(features)[1]):
        #     plt.imshow(np.transpose(data_native[i],(1,2,0)))
        #     plt.savefig("./log/object/8/naive_"+str(i))
        #     # plt.show()
        #     for j in range(np.shape(features)[0]):
        #         m = features[j][i]
        #
        #         plt.imshow(np.transpose(m,(1,2,0)))
        #         plt.title(str(np.sum(m))+' & '+str(compute_local_error(m)))
        #         plt.savefig("./log/object/8/rec_" + str(i)+'_'+str(j))
        #         # plt.show()

        for i in range(len(features)):
            # plot_images_grid(torch.tensor(np.transpose(reconstruction.cpu().numpy(), (0, 3, 1, 2))), export_img=path + '/reconstruction', title='Reconstructions', padding=2)
            plot_images_grid(torch.tensor(reconstructions[i][0::4]), export_img=path + '/Img_reconstruction_'+str(i), title='Reconstructions '+str(i),
                             nrow=4, padding=4)
            plot_images_grid(torch.tensor(features[i][0::4]), export_img=path + '/Img_feature_'+str(i), title='Feature Map '+str(i), nrow=4,
                             padding=4)
            with torch.no_grad():
                np.save('./log/object/' + str(normal_classes) + '/Data_Reconstruction_'+str(i)+'.npy', reconstructions[i].cpu().numpy())
                np.save('./log/object/' + str(normal_classes) + '/Data_Feature_'+str(i)+'.npy', features[i])

def compute_local_error(error, windows_length = 16, stride = 4):
    c, l ,h = np.shape(error)
    error_max = 0
    for i in range(0,l-windows_length, stride):
        for j in range(0, h-windows_length, stride):
            errror_temp = np.sum(error[:,i:i+windows_length,j:j+windows_length])
            if errror_temp> error_max:
                error_max = errror_temp
    return error_max

