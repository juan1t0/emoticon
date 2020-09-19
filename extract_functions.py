import numpy as np

import os
from os.path import exists, join
import json

import torch
from torch.autograd import Variable
import cv2

#######  get the Face landmarks  #######

def getFaceLandmarks(img, flmodel, npoints=68):
  '''
  img : image as numpy array
  flmodel : the model for obtain the landmarks
  npoints : the number of landmarks to detect, 68 by default

  return a 1D numpy array with the landmarks
  '''
  face = flmodel.get_landmarks(img)
  # face = flmodel.get_landmarks_from_image(img)

  if face == None:
    return np.ones(npoints*2)
  # face = face[0]
  # print('landmarks',face[0].shape)
  flm = np.zeros(npoints*2)
  # flm = list()
  for i,lm in enumerate(face[0]):
    flm[i*2] = lm[0]
    flm[(i*2) +1] = lm[1]
    # flm.append(lm[0])
    # flm.append(lm[1])
  
  # flm = np.asarray(flm)
  return flm


#######  get the skeleton pose   #######

def getSkeletonPose(imgFolder, imgFilename, imgIndex, tempFolder='/content/temp'):
  '''
  imgFolder : Folder where images are placed
  imgFilename : name of the image
  imgIndex : images' index #### del
  tempFolder : Folder to save the result

  return the numpy array with the skeleton
  '''
  if not exists(tempFolder):
    os.mkdir(tempFolder)

  imgpath = '--video '+ '/content/emotic/' + imgFolder +'/'+ imgFilename
  imgsave = ' --write_json '+ tempFolder
  
  state = os.system('cd openpose && ./build/examples/openpose/openpose.bin '+ imgpath + imgsave + ' --display 0 --render_pose 0')
  # print(state)
  if state != 0:
      C, T, V, N = 3, 1, 25, 1 #chanels, frame, joints, persons
      return np.zeros((C, T, V, N))

  sample_name = imgFilename[:-4] + '_000000000000_keypoints.json'
  sample_path = os.path.join('./temp', sample_name)

  with open(sample_path, 'r') as f:
    skln = json.load(f)
  
  data = skln['people']
  # print(type(data))

  C, T, V, N = 3, 1, 25, 1 #chanels, frame, joints, persons
  
  data_numpy = np.zeros((C, T, V, N))
  pose = data[0]['pose_keypoints_2d']
  data_numpy[0, 0, :, 0] = pose[0::3]
  data_numpy[1, 0, :, 0] = pose[1::3]
  data_numpy[2, 0, :, 0] = pose[2::3]
  
  rmfn = tempFolder + '/' + sample_name
  os.system('rm ' + rmfn)
  return data_numpy

#######    get the depth mask    #######

def getDepthMask(image, depthmodel, finalW=224, finalH=224):
  '''
  image : numpy array
  depthmodel : Megadepth pretrained model
  finalW & finalH : final dimension

  return the depthmask of image in a numpy array
  '''
  img = np.float32(image)/255.0
  # 384x512 because the depthmodel
  img = cv2.resize(img, (384, 512))#, order = 1)
  input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
  input_img = input_img.unsqueeze(0)
  input_image = Variable(input_img.cuda())

  pred_depth = depthmodel.netG.forward(input_image) 
  pred_depth = torch.squeeze(pred_depth)

  depth = 1/(torch.exp(pred_depth))
  depth = depth.data.cpu().numpy()
  depth = depth/np.amax(depth)
  depth = cv2.resize(depth,(finalW,finalH))
  return depth

#######   get the context mask   #######

def getContextMask(img, strbbox, finalW=224, finalH=224):
  '''
  img : numpy array
  strbbox : str with the bbox, format '[y1,x1,y2,x2]'
  finalW & finalH : final dimension

  return img with zeros intead of bbox
  '''
  bbox = [int(x) for x in (strbbox[1:-1]).split(',')]
  # print('box',bbox)
  x1, x2, y1, y2 = bbox[1], bbox[3], bbox[0], bbox[2]
  if x2 > img.shape[0]:
    x2 = img.shape[0]
  if y2 > img.shape[1]:
    y2 = img.shape[1]

  cm = np.copy(img)
  cm[x1:x2,y1:y2,:] = np.zeros((x2-x1, y2-y1, 3))
  cm = cv2.resize(cm, (finalW,finalH))
  return cm