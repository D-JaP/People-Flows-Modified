import  h5py
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import json
from image import *

#set the root to the path of FDST dataset you download
<<<<<<< HEAD
root = '../data'
=======
root = '../our_dataset'
>>>>>>> mobilecountx2

#now generate the FDST's ground truth
train_folder = os.path.join(root,'train_data1')
test_folder = os.path.join(root,'test_data')
path_sets = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder,f))]+[os.path.join(test_folder,f) for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder,f))]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)



# for data analysis
count = []

for img_path in img_paths:
    print (img_path)
    gt_path = img_path.replace('.jpg','.json')
    with open (gt_path,'r') as f:
        gt = json.load(f)

    anno_list = list(gt.values())[0]['regions']
<<<<<<< HEAD
    img= plt.imread(img_path)
    k = np.zeros((720,1280))
    rate_h = img.shape[0]/720.0
    rate_w = img.shape[1]/1280.0
    for i in range(0,len(anno_list)):
        y_anno = min(int(anno_list[i]['shape_attributes']['y']/rate_h),720)
        x_anno = min(int(anno_list[i]['shape_attributes']['x']/rate_w),1280)
        k[y_anno,x_anno]=1
    k = gaussian_filter(k,3)
    with h5py.File(img_path.replace('.jpg','_resize.h5'), 'w') as hf:
            hf['density'] = k
            hf.close()
=======
    # img= plt.imread(img_path)
    # k = np.zeros((360,640))
    # rate_h = img.shape[0]/360.0
    # rate_w = img.shape[1]/640.0
    # for i in range(0,len(anno_list)):
    #     y_anno = min(int(anno_list[i]['shape_attributes']['y']/rate_h),360)
    #     x_anno = min(int(anno_list[i]['shape_attributes']['x']/rate_w),640)
    #     k[y_anno,x_anno]=1
    # k = gaussian_filter(k,3)
    # with h5py.File(img_path.replace('.jpg','_resize.h5'), 'w') as hf:
    #         hf['density'] = k
    #         hf.close()
    count.append(len(anno_list))

mean = sum(count)/len(count)
print("mean: ", mean)


# Median calculation
sorted_data = sorted(count)
n = len(count)
if n % 2 == 0:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
else:
    median = sorted_data[n // 2]


print("median: ", median)

# Variance calculation
squared_diff = [(x - mean) ** 2 for x in count]
variance = sum(squared_diff) / (len(count) - 1)
print("variance: ", variance)


# # Create a histogram
# plt.figure(figsize=(8, 6))
# plt.hist(count, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
# # plt.axvline(sum(count) / len(count), color='red', linestyle='dashed', linewidth=2, label='Mean')
# # plt.axvline(25, color='green', linestyle='dashed', linewidth=2, label='Median')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram with Mean and Median')
# plt.legend()
# plt.grid(True)

# plt.show()


# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(count, vert=False, labels=['Data'], notch=True, patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.axvline(sum(count) / len(count), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(25, color='green', linestyle='dashed', linewidth=2, label='Median')
plt.xlabel('Value')
plt.title('Box Plot with Mean and Median')
plt.legend()
plt.grid(True)

plt.show()
>>>>>>> mobilecountx2
