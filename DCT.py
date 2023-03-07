import cv2
import numpy as np
import glob, os
from math import log10, sqrt
import argparse
import pdb
from scipy.fftpack import dct, idct


def metric(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# implement 2D DCT
def dct2(a):
    return dct(dct(a, axis = 0, norm='ortho'), axis = 1, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a, axis = 0, norm='ortho'), axis = 1, norm='ortho')    

def generateRandomMask(N, K):
	tmp = np.zeros(N*N)
	for i in range(K):
		tmp[i] = 255
	imgMask = tmp.reshape((N,N))
	return imgMask

def img_dct_fullThres(img_gray, K, N=8):
	# thresholding based on value
	print("In Function")
	percent = 100 - (K * 100./64)
	print(percent)
	reconstructed = np.zeros(img_gray.shape)
	I_thresh = np.zeros((N,N))
	dct = np.zeros(img_gray.shape)
	# Do 8x8 DCT on image (in-place)
	for i in range(0,img_gray.shape[0],N):
		for j in range(0,img_gray.shape[1],N):
			dct[i:(i+N),j:(j+N)] = dct2(img_gray[i:(i+N),j:(j+N)])

	thresh = np.percentile(abs(dct).ravel(), percent)
	I_thresh = dct * (abs(dct) > thresh)
	#print(thresh)
	#I_thresh = dct * (abs(dct) > (thresh*np.max(dct)))
	for i in range(0,img_gray.shape[0],N):
		for j in range(0,img_gray.shape[1],N):			
			#pdb.set_trace()
			reconstructed[i:(i+N),j:(j+N)] = idct2(I_thresh[i:(i+N),j:(j+N)])
	print("Keeping %.2f%% of DCT coefficients"%(100*np.sum(I_thresh != 0.0)/I_thresh.size))
	return reconstructed

def img_dct(img_gray, K, N=8):
	# thresholding based on value
	percent = 100 - (K * 100./64)
	print(100 - percent)
	reconstructed = np.zeros(img_gray.shape)
	I_thresh = np.zeros(img_gray.shape)
	dct = np.zeros(img_gray.shape)
	# Do 8x8 DCT on image (in-place)
	for i in range(0,img_gray.shape[0],N):
		for j in range(0,img_gray.shape[1],N):
			dct[i:(i+N),j:(j+N)] = dct2(img_gray[i:(i+N),j:(j+N)])
			thresh = np.percentile(abs(dct[i:(i+N),j:(j+N)]).ravel(), percent)
			I_thresh[i:(i+N),j:(j+N)] = dct[i:(i+N),j:(j+N)]  * (abs(dct[i:(i+N),j:(j+N)])  > thresh)
			#pdb.set_trace()
			reconstructed[i:(i+N),j:(j+N)] = idct2(I_thresh[i:(i+N),j:(j+N)])
	print("Keeping %.2f%% of DCT coefficients"%(100*np.sum(I_thresh != 0.0)/I_thresh.size))
	return reconstructed

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='project02')
	parser.add_argument('--name', type=str, default="", help = 'file name')
	parser.add_argument('--K', type=int, default=32, help = 'coefficient number')
	args = parser.parse_args()
	path = 'testing/' + args.name
	print(path)
	N = 8
	# put the training data into k 4x4 blocks
	img = cv2.imread(path)
	# convert to grayscale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dct_out = np.zeros(img_gray.shape)
	#generate random mask on coefficient
	#maskCof = generateRandomMask(N, args.K)
	#print(maskCof)
	dct_out = img_dct(img_gray, args.K)
	
	psnr= metric(img_gray, dct_out)
	print("DCT-"+ str(args.K) +", PSNR : " + str(psnr))
	
	if os.path.exists('dct/') != True:
		os.makedirs('dct/')
	cv2.imwrite('dct/' + args.name.split('.')[0] +'_gray.jpg', img_gray)
	cv2.imwrite('dct/' + args.name.split('.')[0] +'_out_'+ str(args.K) + '.jpg', dct_out)
# Example
#python DCT.py --name goldhill.png --K 32