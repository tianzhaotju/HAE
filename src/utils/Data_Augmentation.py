import cv2
import  numpy as np
import os
name_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable','capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
path = '/home/tank/桌面/Deep-SVDD/data/Mvtec/'

i = 9

image_path = path+name_list[i]+'/train/good/'
image_save_path = path+name_list[i]+'/train/good1/'
print(image_save_path)

# read all images
Images = []
for root, dirs, files in os.walk(image_path):
    for file in files:
        file_path = os.path.join(root, file)
        im_tem = cv2.imread(file_path)
        Images.append(im_tem)

Images = np.array(Images)
l, _, _, _ = Images.shape


# # Data augmentation for texture
# iter = int((10000-l)/(l))+1
# for j in range(l):
# #for j in range(2):
#
#     # resize
#     img_native = cv2.resize(Images[j], (512, 512))
#
#     l_rand = np.random.randint(384)
#     h_rand = np.random.randint(384)
#     clip_native = img_native[l_rand:l_rand+128, h_rand:h_rand+128,:]
#     cv2.imwrite(image_save_path + 'native'+ str(j)+'.png', clip_native)
#
#     for i in range(iter):
#     #for i in range(2):
#         # ratation
#         rows, cols = img_native.shape[:2]
#         angle = np.random.random(1)*360-360
#         M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
#         #img_rotation = cv2.warpAffine(img_native, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
#         #img_rotation = cv2.warpAffine(img_native, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE101)
#         img_rotation = cv2.warpAffine(img_native, M, (cols, rows), borderMode=cv2.BORDER_REFLECT101)
#
#
#         l_rand_1 = np.random.randint(384)
#         h_rand_1 = np.random.randint(384)
#         clip_rotation = img_rotation[l_rand_1:l_rand_1 + 128, h_rand_1:h_rand_1 + 128, :]
#         cv2.imwrite(image_save_path + 'rotation' + str(i)+ str(j)+'.png', clip_rotation)
#
#



# Data augmentation for object
iter = int((10000-l)/(l*2))+1
for j in range(l):
#for j in range(2):

    # resize
    img_native = cv2.resize(Images[j], (256, 256))

    cv2.imwrite(image_save_path+str(j)+'native.png',img_native)

    for i in range(iter):
    #for i in range(2):
        # ratation
        rows, cols = img_native.shape[:2]
        angle = np.random.random(1)*360-360
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        #img_rotation = cv2.warpAffine(img_native, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        img_rotation = cv2.warpAffine(img_native, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(image_save_path + str(i) +str(j)+ 'roration.png', img_rotation)
        #cv2.BORDER_REFLECT

        v= np.random.random(1) * 40-40
        h = np.random.random(1) * 40-40
        # translation
        M_trans = np.array([[1, 0, v], [0, 1, h]], dtype=np.float32)
        img_translation = cv2.warpAffine(img_native, M_trans, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(image_save_path + str(i) +str(j)+ 'translation.png', img_translation)



