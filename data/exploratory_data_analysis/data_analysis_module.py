import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def general_remarks_train(x):
    '''x = data.data_train'''
    print('Number of train examples:', x.shape[0])
    sum = 0
    for i in x:
        sum += len(i)
    sum = sum/22050
    print('Total duration of the IRMAS training set:', sum, 'seconds')

def general_remarks_test(x):
    '''x = data.data_test'''
    print('Number of test examples:', x.shape[0])
    sum = 0
    for i in x:
        sum += len(i)
    sum = sum/22050
    print('Total duration of the IRMAS test set:', sum, 'seconds')

def duration_histogram_test(x):
    '''x = data.data_test'''
    duration = []
    for i in x:
        duration.append(len(i)/22050)
    plt.figure(figsize=(12,6))
    plt.hist(duration, bins=20, color='firebrick', rwidth=0.8)
    plt.title('Duration histogram of the IRMAS test data', size=15)
    plt.xlabel('duration in seconds', size=12)
    plt.ylabel('number of examples', size=12)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\duration_histogram_test.png')
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
    plt.xticks(ticks=np.arange(11), labels=labels, size=12)
    plt.title('Duration of the IRMAS training data for each instrument', size=15)
    plt.ylabel('duration in seconds', size=12)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\instrument_duration_train.png')
    plt.show()
    plt.close()

def instrument_duration_test(x, y):
    '''x = data.data_test, y = data.y_test'''
    labels = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
    test_dur = np.zeros(11)
    for i, x in enumerate(x):
        test_dur += y[i]*len(x)/22050
    plt.figure(figsize=(12,6))
    plt.bar(np.arange(11), test_dur, color='darkblue')
    plt.xticks(ticks=np.arange(11), labels=labels, size=12)
    plt.title('Duration of the IRMAS test data for each instrument', size=15)
    plt.ylabel('duration in seconds', size=12)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\instrument_duration_test.png')
    plt.show()
    plt.close()

def instrument_number_test(y):
    '''y = data.y_test'''
    inst_num = np.zeros(11)

    for i in y:
        index = int(np.sum(i)-1)
        inst_num[index] += 1
    
    plt.figure(figsize=(12,6))
    plt.bar(np.arange(1,12), inst_num, color='darkgreen')
    plt.xticks(ticks=np.arange(1,12))
    plt.title('Number of instruments distribution in the IRMAS test data', size=15)
    plt.xlabel('number of instruments', size=12)
    plt.ylabel('number of examples', size=12)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\instrument_number_test.png')
    plt.show()
    plt.close()

def instrument_incidence_correlation(y):
    '''y = data.y_test'''
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    df = pd.DataFrame(data=y, columns=labels)
    corrM = df.corr()
    plt.figure(figsize=(15,15))
    plt.matshow(corrM, fignum=0, cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Instrument incidence correlation matrix for the IRMAS test set', size=15)
    plt.xticks(ticks=np.arange(11), labels=labels, rotation='vertical', size=12)
    plt.yticks(ticks=np.arange(11), labels=labels, size=12)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\instrument_incidence_correlation_test.png')
    plt.show()
    plt.close()

def data_correlation_matrix_train(X, y):
    '''x = data.X_train, y = data.y_train'''
    a = np.concatenate((y, X), axis=1)
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi',
        'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
        'zcr', 'rms', 'sp_roll', 'sp_cent', 'sp_band',
        'sp_cont1', 'sp_cont2', 'sp_cont3', 'sp_cont4', 'sp_cont5', 'sp_cont6', 'sp_cont7',
        'sp_flat', 'perm_ent', 'sp_ent', 'svd_ent', 'Hjorth_mob', 'Hjorth_com']
    df = pd.DataFrame(data=a, columns=labels)
    corrM = df.corr()
    plt.figure(figsize=(15,15))
    plt.matshow(corrM, fignum=0, cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation matrix for instruments and features in the IRMAS training set', size=15)
    plt.xticks(ticks=np.arange(39), labels=labels, rotation='vertical', size=12)
    plt.yticks(ticks=np.arange(39), labels=labels, size=12)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_correlation_matrix_train.png')
    plt.show()
    plt.close()
    print('Top five features with the highest variance in correlation:')
    var = pd.DataFrame.var(corrM)
    var = pd.Series.sort_values(var, ascending= False)
    return var

def data_correlation_matrix_test(X, y):
    '''x = data.X_test, y = data.y_test'''
    a = np.concatenate((y, X), axis=1)
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi',
        'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
        'zcr', 'rms', 'sp_roll', 'sp_cent', 'sp_band',
        'sp_cont1', 'sp_cont2', 'sp_cont3', 'sp_cont4', 'sp_cont5', 'sp_cont6', 'sp_cont7',
        'sp_flat', 'perm_ent', 'sp_ent', 'svd_ent', 'Hjorth_mob', 'Hjorth_com']
    df = pd.DataFrame(data=a, columns=labels)
    corrM = df.corr()
    plt.figure(figsize=(15,15))
    plt.matshow(corrM, fignum=0, cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation matrix for instruments and features in the IRMAS test set', size=15)
    plt.xticks(ticks=np.arange(39), labels=labels, rotation='vertical', size=12)
    plt.yticks(ticks=np.arange(39), labels=labels, size=12)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_correlation_matrix_test.png')
    plt.show()
    plt.close()
    print('Top five features with the highest variance in correlation:')
    var = pd.DataFrame.var(corrM)
    var = pd.Series.sort_values(var, ascending= False)
    return var

def data_clusters_train(X, y):
    '''X=data.X_train, y=data.y_train'''
    X_scaled = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    colors = rainbow(np.linspace(0, 1, 11))
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    def color_label(y):
        for i, val in enumerate(y):
            if(val == 1):
                return colors[i], labels[i]
    plt.figure(figsize=(12,8))
    for x, y in zip(X_pca, y):
        plt.scatter(x[0], x[1], s=10, color=color_label(y)[0], label=color_label(y)[1])
    plt.title('The IRMAS training data clusters', fontsize=16)
    plt.xlabel('$PCA_0$', size=12)
    plt.ylabel('$PCA_1$', size=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.tight_layout()
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_clusters_train.png')
    plt.show()
    plt.close()
    print('Explained variance ratio for the first component:', pca.explained_variance_ratio_[0],
          '\nExplained variance ratio for the second component:', pca.explained_variance_ratio_[1])

def data_clusters_test(X, y):
    '''X=data.X_test, y=data.y_test'''
    X_scaled = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(20, 15))

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
    ax1.set_title('cello', size=15)
    ax1.set_xlabel('$PCA_0$', size=12)
    ax1.set_ylabel('$PCA_1$', size=12)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,1], s=10, cmap='coolwarm')
    ax2.set_title('clarinet', size=15)
    ax2.set_xlabel('$PCA_0$', size=12)
    ax2.set_ylabel('$PCA_1$', size=12)
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,2], s=10, cmap='coolwarm')
    ax3.set_title('flute', size=15)
    ax3.set_xlabel('$PCA_0$', size=12)
    ax3.set_ylabel('$PCA_1$', size=12)
    ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,3], s=10, cmap='coolwarm')
    ax4.set_title('acoustic guitar', size=15)
    ax4.set_xlabel('$PCA_0$', size=12)
    ax4.set_ylabel('$PCA_1$', size=12)
    ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,4], s=10, cmap='coolwarm')
    ax5.set_title('electric guitar', size=15)
    ax5.set_xlabel('$PCA_0$', size=12)
    ax5.set_ylabel('$PCA_1$', size=12)
    ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,5], s=10, cmap='coolwarm')
    ax6.set_title('organ', size=15)
    ax6.set_xlabel('$PCA_0$', size=12)
    ax6.set_ylabel('$PCA_1$', size=12)
    ax7.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,6], s=10, cmap='coolwarm')
    ax7.set_title('piano', size=15)
    ax7.set_xlabel('$PCA_0$', size=12)
    ax7.set_ylabel('$PCA_1$', size=12)
    ax8.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,7], s=10, cmap='coolwarm')
    ax8.set_title('saxophone', size=15)
    ax8.set_xlabel('$PCA_0$', size=12)
    ax8.set_ylabel('$PCA_1$', size=12)
    ax9.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,8], s=10, cmap='coolwarm')
    ax9.set_title('trumpet', size=15)
    ax9.set_xlabel('$PCA_0$', size=12)
    ax9.set_ylabel('$PCA_1$', size=12)
    ax10.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,9], s=10, cmap='coolwarm')
    ax10.set_title('violin', size=15)
    ax10.set_xlabel('$PCA_0$', size=12)
    ax10.set_ylabel('$PCA_1$', size=12)
    ax11.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:,10], s=10, cmap='coolwarm')
    ax11.set_title('voice', size=15)
    ax11.set_xlabel('$PCA_0$', size=12)
    ax11.set_ylabel('$PCA_1$', size=12)
    plt.tight_layout()
    plt.suptitle('The IRMAS test data clusters', fontsize=16, y=1.01)
    plt.savefig(r'.\data\exploratory_data_analysis\figures\data_clusters_test.png')
    plt.show()
    plt.close()
    print('Explained variance ratio for the first component:', pca.explained_variance_ratio_[0],
          '\nExplained variance ratio for the second component:', pca.explained_variance_ratio_[1])