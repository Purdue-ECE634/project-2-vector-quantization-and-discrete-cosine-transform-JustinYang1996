import cv2
import numpy as np
import glob, os
from math import log10, sqrt
import argparse
import pdb

def generateTrainSet(path):
	# initialize training set
	trainSet = np.zeros([1,4,4])
	for img_name in path:
		#print(img_name)
		img = cv2.imread(img_name)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		imgVec = np.zeros([img_gray.shape[0]*img_gray.shape[1]//16, 4, 4])
		count = 0
		for i in range(img_gray.shape[0]//4):
			for j in range(img_gray.shape[1]//4):
				imgVec[count] = img_gray[4*i:4*(i+1), 4*j:4*(j+1)]
				count += 1
		trainSet = np.append(trainSet, imgVec, axis = 0)
	# Delete dummy first row
	trainSet = np.delete(trainSet, 0, axis = 0)
	print("Finish Generate Trainset")
	return (trainSet)

def mse(A, B):
	return np.mean((A - B) ** 2)

def train_codebook(F, L):
	# train the codebook. follow pseudo code at lecture 7, slide 24
	C = np.arange(0,256,256/L).reshape(L,1)
	C = np.tile(C,16).reshape([L,4,4])
	# set initial codebook
	# each block contains 16 (4x4) pixels
	count = F.shape[0]
	T = 0.01
	i = 0
	maxIteration = 100
	D0 = 0
	print("Start Train")
	while i < maxIteration:
		quantE = np.zeros([count])
		codeword = np.zeros([count])
		for j in range(count):
			# for each 4x4 block in training set
			mse_tmp = float('inf')
			c_tmp = 0
			# compute initial MSE
			for l in range(L):
				if mse(F[j], C[l]) < mse_tmp:
					# find minimum mse
					c_tmp = l
					mse_tmp = mse(F[j], C[l])
			# get initial codeword for each block
			quantE[j] = mse_tmp
			codeword[j] = c_tmp
		# Calculate initial distortion
		D1 = np.mean(quantE)
		if D0==0 or np.abs(D1 - D0)/D0 >= T:
			i += 1
			print('iteration '+ str(i) + ', Distortion = ' + str(D1))
			D0 = D1
			#update centroid with mean
			for l in range(L):
				# update codebook
				C[l] = np.mean(F[codeword == l], axis = 0)
		else:
			i += 1
			print('Final iteration '+ str(i) + ', Distortion = ' + str(D1))
			return C, codeword
	return C, codeword


def metric(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def quantize(img, C, codeword):
	# perform vector quantization using trained codebook & corresponding codeword for each block
	print('Quantization')
	quantized = np.zeros(img.shape)
	count = 0
	for i in range(img_gray.shape[0]//4):
		for j in range(img_gray.shape[1]//4):
			# for each 4x4 block
			quantized[4*i:4*(i+1), 4*j:4*(j+1)] = C[int(codeword[count])]
			count += 1

	return quantized

def quantize10(img, C, L):
	# perform vector quantization with 10 image training set
	print('Quantization 10')
	quantized = np.zeros(img.shape)
	for i in range(img.shape[0]//4):
		for j in range(img.shape[1]//4):
			# for each 4x4 block
			block = img[4*i:4*(i+1), 4*j:4*(j+1)]
			mse_tmp = mse(block, C[0])
			c_tmp = 0
			# compute initial MSE
			for l in range(L):
				if mse(block, C[l]) < mse_tmp:
					# find minimum mse
					c_tmp = l
					mse_tmp = mse(block, C[l])
			# quantization
			quantized[4*i:4*(i+1), 4*j:4*(j+1)] = C[c_tmp]
			# reconstruction
	return quantized


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='project02')
	parser.add_argument('--name', type=str, default="", help = 'file name')
	parser.add_argument('--single', type=int, default=1)
	parser.add_argument('--L', type=int, default=128, help = 'quantization level')
	args = parser.parse_args()
	path = 'testing/' + args.name
	if args.single == 1:
		# Use single images for training
		print("Single training")
		training_set_path = path.split()
	else:
		# Use 10 image for training
		print("10 Image Training")
		training_set_path = glob.glob('training/*')
	
	# put the training data into k 4x4 blocks
	training_set = generateTrainSet(training_set_path)
	img = cv2.imread(path)
	# convert to grayscale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	
	if args.single == True:
		codebook, finalCodeword = train_codebook(training_set, args.L)
		quantized = quantize(img_gray, codebook, finalCodeword)
		psnr= metric(img_gray, quantized)
		print("PSNR : " + str(psnr))
		if os.path.exists('VQoutput_single/') != True:
			os.makedirs('VQoutput_single/')
		cv2.imwrite('VQoutput_single/' + args.name.split('.')[0] +'_gray.jpg', img_gray)
		cv2.imwrite('VQoutput_single/' + args.name.split('.')[0] +'_out_'+ str(args.L) + '.jpg', quantized)
	else:
		codebook, finalCodeword = train_codebook(training_set, args.L)
		quantized = quantize10(img_gray, codebook, args.L)
		psnr= metric(img_gray, quantized)
		print("PSNR : " + str(psnr))		
		if os.path.exists('VQoutput_multiple/') != True:
			os.makedirs('VQoutput_multiple/')
		cv2.imwrite('VQoutput_multiple/' + args.name.split('.')[0] +'_gray.jpg', img_gray)
		cv2.imwrite('VQoutput_multiple/' + args.name.split('.')[0] +'_out_'+ str(args.L) + '.jpg', quantized)

#Example
# python VQ.py --name goldhill.png --L 128
