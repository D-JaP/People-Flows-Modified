import h5py
import json
import PIL.Image as Image
import numpy as np
import os
from image import *
from model import CANNet2s
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import os
import math

from matplotlib import cm

from torchvision import transforms


def plotOFlow(x_flow, y_flow, plot_path):
    # Calculate the magnitude and angle (direction) from x and y components
    magnitude, angle = cv2.cartToPolar(x_flow, y_flow, angleInDegrees=True)

    # Normalize the angle to [0, 1]
    normalized_angle = (angle % 360) / 360.0

    # Map normalized angle to hue
    hue =  angle +180

    # Scale magnitude to fit within [0, 1]
    max_magnitude = np.max(magnitude)
    scaled_magnitude = np.clip(magnitude / max_magnitude, 0, 1)

    # Define the filename for the CSV file
    csv_filename = "optical_flow_angles.csv"
    # Normalize the angle to [0, 1]
    normalized_angle = (angle % 360) / 360.0
    # Save the angle data to a CSV file
    np.savetxt(csv_filename, scaled_magnitude, delimiter=',', fmt='%.2f')

    # Set saturation to a constant value (e.g., 1)
    saturation = np.ones_like(hue)

    # Set the value component using the scaled magnitude
    value = np.where(scaled_magnitude < 0.2, 0, scaled_magnitude)
    # Create an HSV image
    hsv_image = np.stack((hue, saturation, value), axis=-1)

    # Convert HSV to BGR (RGB) for visualization
    bgr_image = (cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR) * 255).astype(np.uint8)
    bgr_image = 255 - bgr_image
    cv2.imwrite(plot_path, bgr_image)


def plotDensity(density,plot_path):
    '''
    @density: np array of corresponding density map
    @plot_path: path to save the plot
    '''
    density= density*255.0

    #plot with overlay
    colormap_i = cm.jet(density)[:,:,0:3]

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:,:,0] = overlay_i[:,:,2]
    new_map[:,:,2] = overlay_i[:,:,0]
    cv2.imwrite(plot_path,new_map*255)

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# json file contains the test images
test_json_path = './test.json'

# the folder to output density map and flow maps
output_folder = 'C:/Users/dzung/project/people-flow-modified/People-Flows-Modified/plot'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)



model = CANNet2s()

model = model.cuda()

checkpoint = torch.load('model_best_test_2_26.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []

for i in range(len(img_paths)):
    print(i)
    img_path = img_paths[i].replace('\\', '/')
    
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])

    prev_index = int(max(1,index-5))
    prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')

    prev_img = prev_img.resize((640,360))
    img = img.resize((640,360))

    prev_img = transform(prev_img).cuda()
    img = transform(img).cuda()

    gt_path = img_path.replace('.jpg','_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    prev_img = prev_img.cuda()
    prev_img = Variable(prev_img)

    img = img.cuda()
    img = Variable(img)


    img = img.unsqueeze(0)
    prev_img = prev_img.unsqueeze(0)

    prev_flow = model(prev_img,img)

    prev_flow_inverse = model(img,prev_img)

    mask_boundry = torch.zeros(prev_flow.shape[2:])
    mask_boundry[0,:] = 1.0
    mask_boundry[-1,:] = 1.0
    mask_boundry[:,0] = 1.0
    mask_boundry[:,-1] = 1.0

    mask_boundry = Variable(mask_boundry.cuda())

    reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry

    reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

    overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).data.cpu().numpy()

    base_name = os.path.basename(img_path)
    folder_name = os.path.dirname(img_path).split('/')[-1]

    os.makedirs(os.path.join(output_folder+'/'+folder_name), exist_ok=True)

    gt_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_gt.jpg')
    density_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_pred.jpg')
    flow_1_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_1.jpg')
    flow_2_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_2.jpg')
    flow_3_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_3.jpg')
    flow_4_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_4.jpg')
    flow_5_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_5.jpg')
    flow_6_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_6.jpg')
    flow_7_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_7.jpg')
    flow_8_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_8.jpg')
    flow_9_path = os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_flow_9.jpg')

    pred = cv2.resize(overall,(overall.shape[1]*8,overall.shape[0]*8),interpolation = cv2.INTER_CUBIC)/64.0
    prev_flow= prev_flow.data.cpu().numpy()[0]
    flow_1 = cv2.resize(prev_flow[0],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_2 = cv2.resize(prev_flow[1],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_3 = cv2.resize(prev_flow[2],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_4 = cv2.resize(prev_flow[3],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_5 = cv2.resize(prev_flow[4],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_6 = cv2.resize(prev_flow[5],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_7 = cv2.resize(prev_flow[6],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_8 = cv2.resize(prev_flow[7],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_9 = cv2.resize(prev_flow[8],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    # plotDensity(pred,density_path)
    # plotDensity(target,gt_path)
    # plotDensity(flow_1,flow_1_path)
    # plotDensity(flow_2,flow_2_path)
    # plotDensity(flow_3,flow_3_path)
    # plotDensity(flow_4,flow_4_path)
    # plotDensity(flow_5,flow_5_path)
    # plotDensity(flow_6,flow_6_path)
    # plotDensity(flow_7,flow_7_path)
    # plotDensity(flow_8,flow_8_path)
    # plotDensity(flow_9,flow_9_path)
    
    flow_combine_1 = flow_1 - flow_9
    flow_combine_2 = flow_2 - flow_8
    flow_combine_3 = flow_3 - flow_7
    flow_combine_4 = flow_6 - flow_4
    flow_x = flow_combine_4 + flow_combine_2 - flow_combine_1
    flow_y = flow_combine_2 + flow_combine_1 + flow_combine_3



    plotOFlow(flow_x,flow_y, os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_hsv.jpg'))
    print(os.path.join(output_folder+'/'+folder_name,base_name).replace('.jpg','_'+folder_name+'_hsv.jpg'))




    
