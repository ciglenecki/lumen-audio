import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    plt.savefig(r'.\data\exploratory_data_analysis\figures\duration_histogram_val.png')
    plt.show()
    plt.close()

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
    plt.savefig(r'.\data\exploratory_data_analysis\figures\instrument_duration_train.png')
    plt.show()
    plt.close()

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
    plt.savefig(r'.\data\exploratory_data_analysis\figures\instrument_duration_val.png')
    plt.show()
    plt.close()

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
    plt.savefig(r'.\data\exploratory_data_analysis\figures\instrument_number_val.png')
    plt.show()
    plt.close()

def data_correlation_matrix_train(X, y):
    '''x = X.data_train, y = data.y_train'''
    a = np.concatenate((y, X), axis=1)
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi',
        'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
        'zcr', 'rms', 'sp_roll', 'sp_cent', 'sp_band',
        'sp_cont1', 'sp_cont2', 'sp_cont3', 'sp_cont4', 'sp_cont5', 'sp_cont6', 'sp_cont7',
        'sp_flat', 'perm_ent', 'sp_ent', 'svd_ent', 'Hjorth_mob', 'Hjorth_com']
    df = pd.DataFrame(data=a, columns=labels)
    corrM = df.corr()
    plt.figure(figsize=(15,15))
    plt.matshow(corrM, fignum=0, cmap='coolwarm')
    plt.colorbar()
    plt.title('Correlation matrix for instruments and features in the IRMAS training set')
    plt.xticks(ticks=np.arange(39), labels=labels, rotation='vertical')
    plt.yticks(ticks=np.arange(39), labels=labels)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_correlation_matrix_train.png')
    plt.show()
    plt.close()
    print('Top five features with the highest variance in correlation:')
    var = pd.DataFrame.var(corrM)
    var = pd.Series.sort_values(var, ascending= False)
    return var

def data_correlation_matrix_val(X, y):
    '''x = data.X_val, y = data.y_val'''
    a = np.concatenate((y, X), axis=1)
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi',
        'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
        'zcr', 'rms', 'sp_roll', 'sp_cent', 'sp_band',
        'sp_cont1', 'sp_cont2', 'sp_cont3', 'sp_cont4', 'sp_cont5', 'sp_cont6', 'sp_cont7',
        'sp_flat', 'perm_ent', 'sp_ent', 'svd_ent', 'Hjorth_mob', 'Hjorth_com']
    df = pd.DataFrame(data=a, columns=labels)
    corrM = df.corr()
    plt.figure(figsize=(15,15))
    plt.matshow(corrM, fignum=0, cmap='coolwarm')
    plt.colorbar()
    plt.title('Correlation matrix for instruments and features in the IRMAS validation set')
    plt.xticks(ticks=np.arange(39), labels=labels, rotation='vertical')
    plt.yticks(ticks=np.arange(39), labels=labels)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_correlation_matrix_val.png')
    plt.show()
    plt.close()
    print('Top five features with the highest variance in correlation:')
    var = pd.DataFrame.var(corrM)
    var = pd.Series.sort_values(var, ascending= False)
    return var

def data_clusters_train(X, y):
    '''X=data.X_train, y=data.y_train'''
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)

    plt.figure(figsize=(15,25))

    ax1 = plt.subplot2grid((4, 3), (0, 0))
    ax2 = plt.subplot2grid((4, 3), (0, 1))
    ax3 = plt.subplot2grid((4, 3), (0, 2))
    ax4 = plt.subplot2grid((4, 3), (1, 0))
    ax5 = plt.subplot2grid((4, 3), (1, 1))
    ax6 = plt.subplot2grid((4, 3), (1, 2))
    ax7 = plt.subplot2grid((4, 3), (2, 0))
    ax8 = plt.subplot2grid((4, 3), (2, 1))
    ax9 = plt.subplot2grid((4, 3), (2, 2))
    ax10 = plt.subplot2grid((4, 3), (3, 0))
    ax11 = plt.subplot2grid((4, 3), (3, 1))

    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,0], s=10, cmap='coolwarm')
    ax1.set_title('cello')
    ax1.set_xlabel('$PCA_0$')
    ax1.set_ylabel('$PCA_1$')
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,1], s=10, cmap='coolwarm')
    ax2.set_title('clarinet')
    ax2.set_xlabel('$PCA_0$')
    ax2.set_ylabel('$PCA_1$')
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,2], s=10, cmap='coolwarm')
    ax3.set_title('flute')
    ax3.set_xlabel('$PCA_0$')
    ax3.set_ylabel('$PCA_1$')
    ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,3], s=10, cmap='coolwarm')
    ax4.set_title('acoustic guitar')
    ax4.set_xlabel('$PCA_0$')
    ax4.set_ylabel('$PCA_1$')
    ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,4], s=10, cmap='coolwarm')
    ax5.set_title('electric guitar')
    ax5.set_xlabel('$PCA_0$')
    ax5.set_ylabel('$PCA_1$')
    ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,5], s=10, cmap='coolwarm')
    ax6.set_title('organ')
    ax6.set_xlabel('$PCA_0$')
    ax6.set_ylabel('$PCA_1$')
    ax7.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,6], s=10, cmap='coolwarm')
    ax7.set_title('piano')
    ax7.set_xlabel('$PCA_0$')
    ax7.set_ylabel('$PCA_1$')
    ax8.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,7], s=10, cmap='coolwarm')
    ax8.set_title('saxophone')
    ax8.set_xlabel('$PCA_0$')
    ax8.set_ylabel('$PCA_1$')
    ax9.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,8], s=10, cmap='coolwarm')
    ax9.set_title('trumpet')
    ax9.set_xlabel('$PCA_0$')
    ax9.set_ylabel('$PCA_1$')
    ax10.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,9], s=10, cmap='coolwarm')
    ax10.set_title('violin')
    ax10.set_xlabel('$PCA_0$')
    ax10.set_ylabel('$PCA_1$')
    ax11.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,10], s=10, cmap='coolwarm')
    ax11.set_title('voice')
    ax11.set_xlabel('$PCA_0$')
    ax11.set_ylabel('$PCA_1$')
    plt.tight_layout()
    plt.suptitle('Training data clusters', fontsize=16, y=1.01)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_clusters_train.png')
    plt.show()
    plt.close()

def data_clusters_val(X, y):
    '''X=data.X_val, y=data.y_val'''
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)

    plt.figure(figsize=(15,25), )

    ax1 = plt.subplot2grid((4, 3), (0, 0))
    ax2 = plt.subplot2grid((4, 3), (0, 1))
    ax3 = plt.subplot2grid((4, 3), (0, 2))
    ax4 = plt.subplot2grid((4, 3), (1, 0))
    ax5 = plt.subplot2grid((4, 3), (1, 1))
    ax6 = plt.subplot2grid((4, 3), (1, 2))
    ax7 = plt.subplot2grid((4, 3), (2, 0))
    ax8 = plt.subplot2grid((4, 3), (2, 1))
    ax9 = plt.subplot2grid((4, 3), (2, 2))
    ax10 = plt.subplot2grid((4, 3), (3, 0))
    ax11 = plt.subplot2grid((4, 3), (3, 1))

    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,0], s=10, cmap='coolwarm')
    ax1.set_title('cello')
    ax1.set_xlabel('$PCA_0$')
    ax1.set_ylabel('$PCA_1$')
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,1], s=10, cmap='coolwarm')
    ax2.set_title('clarinet')
    ax2.set_xlabel('$PCA_0$')
    ax2.set_ylabel('$PCA_1$')
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,2], s=10, cmap='coolwarm')
    ax3.set_title('flute')
    ax3.set_xlabel('$PCA_0$')
    ax3.set_ylabel('$PCA_1$')
    ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,3], s=10, cmap='coolwarm')
    ax4.set_title('acoustic guitar')
    ax4.set_xlabel('$PCA_0$')
    ax4.set_ylabel('$PCA_1$')
    ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,4], s=10, cmap='coolwarm')
    ax5.set_title('electric guitar')
    ax5.set_xlabel('$PCA_0$')
    ax5.set_ylabel('$PCA_1$')
    ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,5], s=10, cmap='coolwarm')
    ax6.set_title('organ')
    ax6.set_xlabel('$PCA_0$')
    ax6.set_ylabel('$PCA_1$')
    ax7.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,6], s=10, cmap='coolwarm')
    ax7.set_title('piano')
    ax7.set_xlabel('$PCA_0$')
    ax7.set_ylabel('$PCA_1$')
    ax8.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,7], s=10, cmap='coolwarm')
    ax8.set_title('saxophone')
    ax8.set_xlabel('$PCA_0$')
    ax8.set_ylabel('$PCA_1$')
    ax9.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,8], s=10, cmap='coolwarm')
    ax9.set_title('trumpet')
    ax9.set_xlabel('$PCA_0$')
    ax9.set_ylabel('$PCA_1$')
    ax10.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,9], s=10, cmap='coolwarm')
    ax10.set_title('violin')
    ax10.set_xlabel('$PCA_0$')
    ax10.set_ylabel('$PCA_1$')
    ax11.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,10], s=10, cmap='coolwarm')
    ax11.set_title('voice')
    ax11.set_xlabel('$PCA_0$')
    ax11.set_ylabel('$PCA_1$')
    plt.tight_layout()
    plt.suptitle('Validation data clusters', fontsize=16, y=1.01)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_clusters_val.png')
    plt.show()
    plt.close()