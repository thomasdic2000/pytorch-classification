import xml.etree.ElementTree as ET
from lxml import etree
from PIL import Image
import numpy as np
import argparse
import glob,os
import re
import sys
import os.path
import matplotlib.pyplot as plt

# function to check given coordinate lies within polygon
def point_in_poly(x,y,poly):
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

def get_imgbw(img):
	imgbw=np.zeros((img.size[0],img.size[1]))
	for i in range(xmin,xmax):
		for j in range(ymin,ymax):
			if point_in_poly(int(i),int(j),coordinate):
				imgbw[i,j]=1
	return imgbw

def extract_patches(images_list, labels_lst, results_dir, patch_h, patch_w, stride_h, stride_w, npatches):
	for i in range(0,len(images_list)):
		filename = images_list[i].split('/')[1].split('.')[0]
		annot_file = labels_lst[i]
		tree = ET.parse(annot_file)
		root = tree.getroot()
		img = Image.open(images_list[i])
		count=0
		for objects in root.iter('object'):
			material = str(objects.find('name').text)
			print filename, material
			xc = []
			yc = []
			coordinate=[]
			count=count+1
			print count
			
			for polygon in objects:
				for xy in polygon.findall('pt'):
                                	x=xy.find('x').text
                                        y=xy.find('y').text
                                        coordinate.append((int(x),int(y)))
                                        xc.append(int(x))
                                        yc.append(int(y))
			try:
				xmin=np.min(xc)
				xmax=np.max(xc)
				ymin=np.min(yc)
				ymax=np.max(yc)
				# cropping image
				img2=img.crop((xmin,ymin,xmax,ymax))
				# creating an array in which coordinates inside polygon equal to 1 and background to 0
				imgbw=np.zeros((img.size[0],img.size[1]))
				for i in range(xmin,xmax):
					for j in range(ymin,ymax):
						if point_in_poly(int(i),int(j),coordinate):
							imgbw[i,j]=1
				#plt.imshow(imgbw)
				#plt.show()
				
				num=0
				# saving the patches
				for i in range(0,(xmax-xmin)/stride_w):
					for j in range(0,(ymax-ymin)/stride_h):
						if num<int(npatches):
							# calculating coordinates of the patch
							pointx_min=xmin+j*stride_w
							pointy_min=ymin+i*stride_h
							pointx_max=pointx_min+patch_w
							pointy_max=pointy_min+patch_h
							#foregroundPixelCount=patch_w*patch_h*((100-tolerance)/100)
							# saving the patch if it meets all the conditions
							#if npatches > 0:
							if np.sum(imgbw[pointx_min:pointx_max,pointy_min:pointy_max]) >=75*75:
								img3=img.crop((xmin+j*stride_h,ymin+i*stride_w,xmin+j*stride_h+patch_h,ymin+i*stride_w+patch_w))
								#plt.imshow(img3)
								#plt.show()
						 		outfile=str(results_dir)+"/"+ str(material) + "/" + str(filename) + "-" + str(material) +str(count)+str(i+j)+".jpg"
								print outfile, material
								img3.save(outfile)
								num=num+1
			except:
				pass

# parse arguments
def parse_args():
        """Parse input arguments"""
        parser = argparse.ArgumentParser(description='Create data')
        parser.add_argument('--img_path',dest='images_dir',help='Input Images directory',type=str)
	parser.add_argument('--annot_path',dest='annot_dir',help='Input Annotations directory',type=str)
        parser.add_argument('--dst',dest='results_dir',help='Results directory',type=str)
        parser.add_argument('--p_h',dest='patch_h',help='Patch height',type=int)
        parser.add_argument('--p_w',dest='patch_w',help='Patch width',type=int)
        parser.add_argument('--stride_h',dest='stride_h',help='Patch height',type=int)
        parser.add_argument('--stride_w',dest='stride_w',help='Patch width',type=int)
	parser.add_argument('--npathces',dest='npatches',help='Number of patches',type=str)
        args = parser.parse_args()
        return args

# getfileList function create list of filenames
def getfileLists(annot_dir, images_dir):
	img_lst = []
	labels_lst = []
	for fname in os.listdir(annot_dir):
		print fname
		file_name = fname.split('.')[0]
		image_file = str(images_dir) + "/" + str(file_name) + ".jpg"
		annatation_file = str(annot_dir) + "/" + str(fname)
		#print image_file, annatation_file
		if os.path.isfile(image_file) and os.path.isfile(annatation_file):
			img_lst.append(str(image_file))
			labels_lst.append(str(annatation_file))
	#print len(img_lst), len(labels_lst)
	return img_lst, labels_lst
	
def main():
        args = parse_args()
	print args
	# get the list of files
	images_list, labels_lst = getfileLists(args.annot_dir, args.images_dir)
	#print len(images_list), len(labels_lst)
	extract_patches(images_list, labels_lst, args.results_dir, args.patch_h, args.patch_w, args.stride_h, args.stride_w,args.npatches)
        #train_imgs,test_imgs = getfileLists(args.input_dir)

if __name__ == '__main__':
        main()

