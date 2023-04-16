import numpy as np
import pandas as pd
import librosa as lb
import antropy as ant
import matplotlib.pyplot as plt

def general_remarks_train(x):
    '''x = data.data_train'''
    print('Number of train examples:', x.shape[0])
    sum = 0
    for i in x:
        sum += len(i)
    sum = sum/22050
    print('Total duration of the training set:', sum, 'seconds')

def general_remarks_val(x):
    '''x = data.data_val'''
    print('Number of validation examples:', x.shape[0])
    sum = 0
    for i in x:
        sum += len(i)
    sum = sum/22050
    print('Total duration of the validation set:', sum, 'seconds')

def duration_histogram_val(x):
    '''x = data.data_val'''
    duration = []
    for i in x:
        duration.append(len(i)/22050)
    plt.figure(figsize=(12,6))
    plt.hist(duration, bins=20, color='firebrick', rwidth=0.8)
    plt.title('Validation data')
    plt.xlabel('duration in seconds')
    plt.ylabel('number of examples')

def instrument_duration_train(x, y):
    '''x = data.data_train, y = data.y_train'''
    labels = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
    train_dur = np.zeros(11)
    for i, x in enumerate(x):
        train_dur += y[i]*len(x)/22050
    plt.figure(figsize=(12,6))
    plt.bar(np.arange(11), train_dur, color='darkblue')
    plt.xticks(ticks=np.arange(11), labels=labels)
    plt.title('Training data')
    plt.ylabel('duration in seconds')

def instrument_duration_val(x, y):
    '''x = data.data_val, y = data.y_val'''
    labels = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
    val_dur = np.zeros(11)
    for i, x in enumerate(x):
        val_dur += y[i]*len(x)/22050
    plt.figure(figsize=(12,6))
    plt.bar(np.arange(11), val_dur, color='darkblue')
    plt.xticks(ticks=np.arange(11), labels=labels)
    plt.title('Validation data')
    plt.ylabel('duration in seconds')

def instrument_number_val(y):
    '''y = data.y_val'''
    inst_num = np.zeros(11)

    for i in y:
        index = int(np.sum(i)-1)
        inst_num[index] += 1
    
    plt.figure(figsize=(12,6))
    plt.bar(np.arange(1,12), inst_num, color='darkgreen')
    plt.xticks(ticks=np.arange(1,12))
    plt.title('Validation data')
    plt.xlabel('number of instruments')
    plt.ylabel('number of examples')

