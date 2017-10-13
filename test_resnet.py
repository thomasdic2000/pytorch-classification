import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#import torch.nn.DataParallel
import os,glob
import skimage
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import OrderedDict
new_state_dict = OrderedDict()
import time
model = models.__dict__['resnet34']()
#model = torch.load('model_best.pth.tar')
checkpoint = torch.load('model_best.pth.tar')  

print checkpoint
state_dict = checkpoint['state_dict']
print('loaded state dict:', state_dict.keys())
model.eval()
model.cuda()
model = torch.nn.DataParallel(model)

for k, v in state_dict.items():                                                                                             
    name = k[7:] # remove `module.`                                                                               
    print name
    if k[0] == 'f':                                                                                                         
        new_state_dict[name] = v                                                                                            
    else:                                                                                                                   
        new_state_dict[k] = v 
model.load_state_dict(new_state_dict)
correct=0
wrong=0
Arr = ['AS','CS','Metal','Tile','Wood']

dir="val"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
labels = {0:'AS',1:'CS',2:'Metal',3:'Tile',4:'Wood'}

def getPerformanceReports(ground_truth,predictions,labels):
	Arr= []
	for i in range(0,len(labels)):
		Arr.append(labels[i])
	print confusion_matrix(ground_truth,predictions)
	print(classification_report(ground_truth,predictions, target_names=Arr))

def computePerformance(model,images,ground_truth):
	predictions = []
	for i in range(0,len(images)):
		#IMAGE_FILE=os.path.abspath(images[i])
		img = Image.open(images[i]).convert('RGB')
        	preprocess = transforms.Compose([transforms.Scale(100),
                     transforms.ToTensor(),
                     normalize])  
		img_tensor = preprocess(img)
		img_tensor.unsqueeze_(0)
		start_time = time.time()
                img_variable = model(torch.autograd.Variable(img_tensor))
		print("--- %s seconds ---" % (time.time() - start_time))
                predictions.append(np.argmax(img_variable.cpu().data.numpy()))
                #print labels[int(ground_truth[i])],labels[int(np.argmax(img_variable.cpu().data.numpy()))]
	return predictions

def getList(dirPath='val'):
	fpath = []
	labels = []
	count=0
	for path, dirs, files in os.walk(dirPath):
		for d in dirs:
			for f in glob.iglob(os.path.join(path, d, '*')):
				count = count +1
				#filename = str(dirPath) + "/" + str(d) + "/" + str(f)
				fpath.append(str(f))
				labels.append(int(d))		
	return fpath,labels

def main():
	global labels, normalize,Arr
	images, ground_truth = getList()
	predictions = computePerformance(model,images, ground_truth)
	getPerformanceReports(ground_truth,predictions,labels)

if __name__ == "__main__":
    main()
