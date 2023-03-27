import glob
import numpy as np
import librosa as lb
from sklearn.svm import SVC, LinearSVC # we can use LinearSVC because it's faster
from sklearn.base import BaseEstimator

class ManageData():

    def __init__(self, ):
        self.data_train = []
        self.y_train = []
        self.data_val = []
        self.y_val = []

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
    
    def LoadTrainData(self, path = r'C:\Users\dragu\Desktop\Lumen\Dataset\Dataset\IRMAS_Training_Data\*\*.wav', label_position = 65):
        """Loads data and labels for training"""

        X = []
        y = []

        for filename in glob.glob(path):
            data, _ = lb.load(filename)
            X.append(data)
            y.append(self._MakeLabel(filename[label_position:(label_position+3)]))

        self.data_train = np.array(X)
        self.y_train = np.array(y)

        print('Training data loaded!\nExample:\nX:', self.data_train[0], '\ny:', self.y_train[0])
    
    def _BreakSignal(self, signal):
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
    
    def LoadValidationData(self, path = r'C:\Users\dragu\Desktop\Lumen\Dataset\Dataset\IRMAS_Validation_Data\set1\*.wav'):
        """Loads the validation data as well as the corresponding labels"""

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
            
            #chunking the data into 3-second chunks
            for chunk in self._BreakSignal(data):
                X.append(chunk)
                y.append(label)
            
        self.data_val = np.array(X)
        self.y_val = np.array(y)

        print('Validation data loaded!\nExample:\nX:', self.data_val[0], '\ny:', self.y_val[0])
    
    def MakeFeatures(self, ): #here we can add more features
        """Makes all the desired features from the loaded data and flattens it into a set of concatenated vectors."""

        features_train = []
        features_val = []

        for i in self.data_train:
            feature1 = lb.feature.melspectrogram(y=i, sr=22050).flatten()
            feature2 = lb.feature.mfcc(y=i, sr=22050).flatten()
            feature3 = lb.feature.rms(y=i).flatten()
            features_train.append(np.concatenate((feature1, feature2, feature3)))

        for i in self.data_val:
            feature1 = lb.feature.melspectrogram(y=i, sr=22050).flatten()
            feature2 = lb.feature.mfcc(y=i, sr=22050).flatten()
            feature3 = lb.feature.rms(y=i).flatten()
            features_val.append(np.concatenate((feature1, feature2, feature3)))

        self.X_train = np.array(features_train)
        self.X_val = np.array(features_val)

        print('Features made!\nExample:\nX_train:', self.X_train[0],'X_val:\n', self.X_val[0])
    
class multilabelSVM(BaseEstimator):

    def __init__(self, model_ = LinearSVC()):
         self.model_ = model_
    
    def fit(self, X = None, y = None):
        """Training a full model and returning a one-hot encoded vector which tells which instrument is present in the given data"""
        X = self.X_train if X is None else X
        y = self.y_train if y is None else y

        self.model0 = LinearSVC()
        self.model0.fit(X = X, y = y[:,0])
        self.model1 = self.model_
        self.model1.fit(X = X, y = y[:,1])
        self.model2 = self.model_
        self.model2.fit(X = X, y = y[:,2])
        self.model3 = self.model_
        self.model3.fit(X = X, y = y[:,3])
        self.model4 = self.model_
        self.model4.fit(X = X, y = y[:,4])
        self.model5 = self.model_
        self.model5.fit(X = X, y = y[:,5])
        self.model6 = self.model_
        self.model6.fit(X = X, y = y[:,6])
        self.model7 = self.model_
        self.model7.fit(X = X, y = y[:,7])
        self.model8 = self.model_
        self.model8.fit(X = X, y = y[:,8])
        self.model9 = self.model_
        self.model9.fit(X = X, y = y[:,9])
        self.model10 = self.model_
        self.model10.fit(X = X, y = y[:,10])
        
        print('Training completed successfully!')
        return self
    
    def predict(self, X = None):
        """Gives a prediction for input X based on a model"""
        X = self.X_val if X is None else X

        prediction = np.array([self.model0.predict(X), self.model1.predict(X), self.model2.predict(X),
                      self.model3.predict(X), self.model4.predict(X), self.model5.predict(X),
                      self.model6.predict(X), self.model7.predict(X), self.model8.predict(X),
                      self.model9.predict(X), self.model10.predict(X)])
        
        prediction = [prediction[:,i] for i in range(prediction.shape[1])]

        return np.array(prediction)