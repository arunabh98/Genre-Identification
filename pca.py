import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os   
import sys

np.set_printoptions(threshold=np.nan)

#File name is the first string after python pca.py <filename.csv>, import.
path = str(sys.argv[1])   
dataset = pd.read_csv(path, header=None, names=['genre', 'track_id', 'artist_name', 'title', 'loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12', 'max_segment_timbre1', 'max_segment_timbre2', 'max_segment_timbre3', 'max_segment_timbre4', 'max_segment_timbre5', 'max_segment_timbre6', 'max_segment_timbre7', 'max_segment_timbre8', 'max_segment_timbre9', 'max_segment_timbre10', 'max_segment_timbre11', 'max_segment_timbre12'
])  

#choose the features. set y as the genre. convert to matrix.
features = ['loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12', 'max_segment_timbre1', 'max_segment_timbre2', 'max_segment_timbre3', 'max_segment_timbre4', 'max_segment_timbre5', 'max_segment_timbre6', 'max_segment_timbre7', 'max_segment_timbre8', 'max_segment_timbre9', 'max_segment_timbre10', 'max_segment_timbre11', 'max_segment_timbre12']
x = dataset.loc[1:, features].values
y = dataset.loc[1:,['genre']].values

#use datascaler from sklearn library which scales the date by subtracting the mean
x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = features).head()

#two component PCA, using sklearn library
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principaldataset = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finaldataset = pd.concat([principaldataset, dataset[['genre']]], axis = 1)

#plotting the two component analysis
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = ['classic pop and rock', 'pop', 'soul and reggae', 'punk', 'folk', 'dance and electronica', 'metal', 'jazz and blues', 'classical', 'hip-hop' ]
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#afeeee', 'burlywood', 'chartreuse']
for target, color in zip(targets,colors):
    indicesToKeep = finaldataset['genre'] == target
    ax.scatter(finaldataset.loc[indicesToKeep, 'principal component 1']
               , finaldataset.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()







