''' 
Model for common spatial pattern (CSP) feature calculation and classification for EEG data
'''

import numpy as np
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold

# import self defined functions 
from csp import generate_projection,generate_eye,extract_feature
from get_data import get_data
from filters import load_filterbank 

class CSP_Model:

	def __init__(self):
		self.data_path 	= 'dataset/'
		self.NO_splits = 5      # number of folds in cross validation 
		self.fs = 250.          # sampling frequency 
		self.NO_channels = 22   # number of EEG channels 
		self.NO_subjects = 9
		self.NO_csp = 24        # Total number of CSP feature per band and timewindow
		self.bw = np.array([2,4,8,16,32]) # bandwidth of filtered signals 
		self.ftype = 'butter' # 'fir', 'butter'
		self.forder= 2 # 4
		self.filter_bank = load_filterbank(self.bw,self.fs,order=self.forder,max_freq=40,ftype = self.ftype) # get filterbank coeffs  
		time_windows_flt = np.array([
		 						[2.5,3.5],
		 						[3,4],
		 						[3.5,4.5],
		 						[4,5],
		 						[4.5,5.5],
		 						[5,6],
		 						[2.5,4.5],
		 						[3,5],
		 						[3.5,5.5],
		 						[4,6],
		 						[2.5,6]]) * self.fs 

		self.time_windows = time_windows_flt.astype(int)
		self.NO_bands = self.filter_bank.shape[0]
		self.NO_time_windows = int(self.time_windows.size/2)
		self.NO_features = self.NO_csp * self.NO_bands * self.NO_time_windows
		
	def extract_csp(self):
		
		w = generate_projection(self.train_data,self.train_label, self.NO_csp,self.filter_bank,self.time_windows)
		feature_mat = extract_feature(self.train_data, w, self.filter_bank, self.time_windows)
		eval_feature_mat = extract_feature(self.eval_data,w,self.filter_bank,self.time_windows)
		
		np.save(str(self.subject) + '-test.npy', eval_feature_mat)
		np.save(str(self.subject) + '-train.npy', feature_mat)
		return 0


	def load_data(self):		
		self.train_data,self.train_label = get_data(self.subject,True,self.data_path)
		self.eval_data,self.eval_label = get_data(self.subject,False,self.data_path)



def main():
	model = CSP_Model()
	print("Number of used features: " + str(model.NO_features))

	# Go through all subjects 
	for model.subject in range(1, model.NO_subjects + 1):  
		print("Subject" + str(model.subject)+":")
		model.load_data()
		model.extract_csp()


if __name__ == '__main__':
	main()
