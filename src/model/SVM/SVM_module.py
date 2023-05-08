import glob
import numpy as np
import librosa as lb
import antropy as ant
from sklearn.svm import SVC, LinearSVC # we can use LinearSVC because it's faster
from sklearn.base import BaseEstimator

class ManageData():

    def __init__(self, ):
        pass

    def LoadTrainData(self, path = r'C:\Users\dragu\Desktop\Lumen\Dataset\Dataset\IRMAS_Training_Data\*\*.wav', label_position = 65):
        """Loads IRMAS data and labels for training"""

        X = []
        y = []

        for filename in glob.glob(path):
            data, _ = lb.load(filename)
            X.append(data)
            y.append(self._MakeLabel(filename[label_position:(label_position+3)]))

        self.data_train = np.array(X)
        self.y_train = np.array(y)

        print('Training data and labels loaded!\nExample:\nX:', self.data_train[0], '\ny:', self.y_train[0])
    
    def LoadTestData(self, path = r'C:\Users\dragu\Desktop\Lumen\Dataset\Dataset\IRMAS_Test_Data\set1\*.wav'):
        """Loads IRMAS test data as well as the corresponding labels"""

        X = []
        y = []

        for filename in glob.glob(path):
            data, _ = lb.load(filename)

            file = open(filename[:-3] + 'txt', "r")
            instruments = file.read()
            instrument_list = instruments.split("\t\n")
            instrument_list.remove('')
            file.close()

            #making a one-hot encoded label of a file
            label = np.zeros(11)
            for instrument in instrument_list:
                single_label = self._MakeLabel(instrument)
                label += single_label
            
            X.append(data)
            y.append(label)
            
        self.data_test = np.array(X, dtype=object)
        self.y_test = np.array(y)

        print('Test data and labels loaded!\nExample:\nX:', self.data_test[0], '\ny:', self.y_test[0])

    def LoadTrainFeatures(self, path = r'C:\Users\dragu\Desktop\Lumen\Dataset\Dataset\IRMAS_Training_Data\*\*.wav', label_position = 65):
        """Loads IRMAS training features and the corresponding labels"""

        X = []
        y = []

        for filename in glob.glob(path):
            data, _ = lb.load(filename)
            X.append(self._Feature(data))
            y.append(self._MakeLabel(filename[label_position:(label_position+3)]))

        self.X_train = np.array(X)
        self.y_train = np.array(y)

        print('Training features and labels loaded!\nExample:\nX:', self.X_train[0], '\ny:', self.y_train[0])
    
    def LoadTestFeatures(self, path = r'C:\Users\dragu\Desktop\Lumen\Dataset\Dataset\IRMAS_Test_Data\set1\*.wav'):
        """Loads IRMAS test features as well as the corresponding labels"""

        X = []
        y = []

        for filename in glob.glob(path):
            data, _ = lb.load(filename)

            file = open(filename[:-3] + 'txt', "r")
            instruments = file.read()
            instrument_list = instruments.split("\t\n")
            instrument_list.remove('')
            file.close()

            #making a one-hot encoded label of a file
            label = np.zeros(11)
            for instrument in instrument_list:
                single_label = self._MakeLabel(instrument)
                label += single_label
            
            X.append(self._Feature(data))
            y.append(label)
            
        self.X_test = np.array(X)#, dtype=object)
        self.y_test = np.array(y)

        print('Test features and labels loaded!\nExample:\nX:', self.X_test[0], '\ny:', self.y_test[0])

    def _MakeLabel(self, instrument):
        '''Makes a one-hot encoded label from the strings describing the instruments contained in the file'''
        
        y = np.zeros(11)

        if instrument == 'cel':
            y[0] = 1
        elif instrument == 'cla':
            y[1] = 1
        elif instrument == 'flu':
            y[2] = 1
        elif instrument == 'gac':
            y[3] = 1
        elif instrument == 'gel':
            y[4] = 1
        elif instrument == 'org':
            y[5] = 1
        elif instrument == 'pia':
            y[6] = 1
        elif instrument == 'sax':
            y[7] = 1
        elif instrument == 'tru':
            y[8] = 1
        elif instrument == 'vio':
            y[9] = 1
        elif instrument == 'voi':
            y[10] = 1
        
        return y

    def _BreakSignal(self, signal): # not used in this model
        chunk_size = 66150
        num_chunks = len(signal) // chunk_size
        remainder = len(signal) % chunk_size

        chunks = [signal[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

        if remainder > 0:
            last_chunk = signal[num_chunks*chunk_size:]
            padding = [last_chunk[-1]] * (chunk_size - remainder)
            last_chunk = np.hstack([last_chunk, padding])
            chunks.append(last_chunk)
    
        return chunks

    def _Feature(self, i):  #here we can add more features
        '''Makes a feature vector from the given raw data'''
        # Mel frequency cepstral coefficients
        mfcc = lb.feature.mfcc(y=i, sr=22050, n_mfcc=10, n_fft=len(i), hop_length=len(i)+1).flatten()
        # Zero crossing rate
        zero_crossing_rate = lb.feature.zero_crossing_rate(y=i, frame_length=len(i), hop_length=len(i)+1).flatten()
        # Root mean square
        rms = lb.feature.rms(y=i, frame_length=len(i), hop_length=len(i)+1).flatten()
        # Roll-off frequency
        spectral_rolloff = lb.feature.spectral_rolloff(y=i, n_fft=len(i), hop_length=len(i)+1).flatten()
        # Spectral centroid
        spectral_centroid = lb.feature.spectral_centroid(y=i, n_fft=len(i), hop_length=len(i)+1).flatten()
        # Spectral bandwidth
        spectral_bandwidth = lb.feature.spectral_bandwidth(y=i, n_fft=len(i), hop_length=len(i)+1).flatten()
        # Spectral contrast
        spectral_contrast = lb.feature.spectral_contrast(y=i, n_fft=len(i), hop_length=len(i)+1).flatten()
        # Spectral flatness
        spectral_flatness = lb.feature.spectral_flatness(y=i, n_fft=len(i), hop_length=len(i)+1).flatten()
        # Permutation entropy
        perm_entropy = ant.perm_entropy(i, normalize=True)
        # Spectral entropy
        spectral_entropy = ant.spectral_entropy(i, sf=22050, method='welch', normalize=True)
        # Singular value decomposition entropy
        svd_entropy = ant.svd_entropy(i, normalize=True)
        # Approximate entropy
        #app_entropy = ant.app_entropy(i)
        # Sample entropy
        #sample_entropy = ant.sample_entropy(i)
        # Hjorth mobility and complexity parameters
        Hjorth_parameters = ant.hjorth_params(i)
        
        return np.hstack((mfcc, zero_crossing_rate, rms, spectral_rolloff, spectral_centroid, spectral_bandwidth,
        spectral_contrast, spectral_flatness, perm_entropy, spectral_entropy, svd_entropy,
        #app_entropy, sample_entropy, Hjorth_parameters))
        Hjorth_parameters))
    
class multilabelSVM_linear(BaseEstimator):

    def __init__(self, C=1):
        self.C = C
    
    def fit(self, X = None, y = None):
        """Training a full model and returning a one-hot encoded vector which tells which instrument is present in the given data"""
        X = self.X_train if X is None else X
        y = self.y_train if y is None else y

        self.model0 = LinearSVC(C=self.C, dual=False)
        self.model0.fit(X = X, y = y[:,0])
        self.model1 = LinearSVC(C=self.C, dual=False)
        self.model1.fit(X = X, y = y[:,1])
        self.model2 = LinearSVC(C=self.C, dual=False)
        self.model2.fit(X = X, y = y[:,2])
        self.model3 = LinearSVC(C=self.C, dual=False)
        self.model3.fit(X = X, y = y[:,3])
        self.model4 = LinearSVC(C=self.C, dual=False)
        self.model4.fit(X = X, y = y[:,4])
        self.model5 = LinearSVC(C=self.C, dual=False)
        self.model5.fit(X = X, y = y[:,5])
        self.model6 = LinearSVC(C=self.C, dual=False)
        self.model6.fit(X = X, y = y[:,6])
        self.model7 = LinearSVC(C=self.C, dual=False)
        self.model7.fit(X = X, y = y[:,7])
        self.model8 = LinearSVC(C=self.C, dual=False)
        self.model8.fit(X = X, y = y[:,8])
        self.model9 = LinearSVC(C=self.C, dual=False)
        self.model9.fit(X = X, y = y[:,9])
        self.model10 = LinearSVC(C=self.C, dual=False)
        self.model10.fit(X = X, y = y[:,10])
        
        print('Training completed successfully!')
        return self
    
    def predict(self, X = None):
        """Gives a prediction for input X based on a model"""
        X = self.X_test if X is None else X

        prediction = np.array([self.model0.predict(X), self.model1.predict(X), self.model2.predict(X),
                      self.model3.predict(X), self.model4.predict(X), self.model5.predict(X),
                      self.model6.predict(X), self.model7.predict(X), self.model8.predict(X),
                      self.model9.predict(X), self.model10.predict(X)])
        
        prediction = [prediction[:,i] for i in range(prediction.shape[1])]

        return np.array(prediction)
    
class multilabelSVM(BaseEstimator):

    def __init__(self, C=1, kernel='rbf'):
        self.C = C
        self.kernel = kernel
    
    def fit(self, X = None, y = None):
        """Training a full model and returning a one-hot encoded vector which tells which instrument is present in the given data"""
        X = self.X_train if X is None else X
        y = self.y_train if y is None else y

        self.model0 = SVC(C=self.C, kernel=self.kernel)
        self.model0.fit(X = X, y = y[:,0])
        self.model1 = SVC(C=self.C, kernel=self.kernel)
        self.model1.fit(X = X, y = y[:,1])
        self.model2 = SVC(C=self.C, kernel=self.kernel)
        self.model2.fit(X = X, y = y[:,2])
        self.model3 = SVC(C=self.C, kernel=self.kernel)
        self.model3.fit(X = X, y = y[:,3])
        self.model4 = SVC(C=self.C, kernel=self.kernel)
        self.model4.fit(X = X, y = y[:,4])
        self.model5 = SVC(C=self.C, kernel=self.kernel)
        self.model5.fit(X = X, y = y[:,5])
        self.model6 = SVC(C=self.C, kernel=self.kernel)
        self.model6.fit(X = X, y = y[:,6])
        self.model7 = SVC(C=self.C, kernel=self.kernel)
        self.model7.fit(X = X, y = y[:,7])
        self.model8 = SVC(C=self.C, kernel=self.kernel)
        self.model8.fit(X = X, y = y[:,8])
        self.model9 = SVC(C=self.C, kernel=self.kernel)
        self.model9.fit(X = X, y = y[:,9])
        self.model10 = SVC(C=self.C, kernel=self.kernel)
        self.model10.fit(X = X, y = y[:,10])
        
        print('Training completed successfully!')
        return self
    
    def predict(self, X = None):
        """Gives a prediction for input X based on a model"""
        X = self.X_test if X is None else X

        prediction = np.array([self.model0.predict(X), self.model1.predict(X), self.model2.predict(X),
                      self.model3.predict(X), self.model4.predict(X), self.model5.predict(X),
                      self.model6.predict(X), self.model7.predict(X), self.model8.predict(X),
                      self.model9.predict(X), self.model10.predict(X)])
        
        prediction = [prediction[:,i] for i in range(prediction.shape[1])]

        return np.array(prediction)