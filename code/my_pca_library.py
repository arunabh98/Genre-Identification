import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os   
import sys
from sklearn import preprocessing
from sklearn.metrics.cluster import silhouette_score

def my_pca(components):

	np.set_printoptions(threshold=np.nan)

	#File name is the first string after python pca.py <filename.csv>, import.
	# path = str(sys.argv[1])   
	dataset = pd.read_csv('../datasets/msd.csv', header=None, names=['genre', 'track_id', 'artist_name', 'title', 'loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12', 'max_segment_timbre1', 'max_segment_timbre2', 'max_segment_timbre3', 'max_segment_timbre4', 'max_segment_timbre5', 'max_segment_timbre6', 'max_segment_timbre7', 'max_segment_timbre8', 'max_segment_timbre9', 'max_segment_timbre10', 'max_segment_timbre11', 'max_segment_timbre12'
	])

	#choose the features. set y as the genre. convert to matrix.
	# features = ['loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12', 'max_segment_timbre1', 'max_segment_timbre2', 'max_segment_timbre3', 'max_segment_timbre4', 'max_segment_timbre5', 'max_segment_timbre6', 'max_segment_timbre7', 'max_segment_timbre8', 'max_segment_timbre9', 'max_segment_timbre10', 'max_segment_timbre11', 'max_segment_timbre12']
	features = ['loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12']
	z = dataset.loc[1:, features]

	# Some columns have string values in numerical columns.
	# So cleaning that.
	for i in range(1, z.shape[1]):
		z = z[pd.to_numeric(z[z.columns[i]], errors='coerce').notnull()]

	# Convert string values to float.
	z = z.convert_objects(convert_numeric=True)

	# LabelEncoder - assigns number 1-n
	le = preprocessing.LabelEncoder()

	# Extract the numerical features and perform normalization.
	reader_numerical = z.select_dtypes(exclude=[object])
	reader_numerical = np.matrix(reader_numerical).astype("double")
	reader_numerical = preprocessing.normalize(reader_numerical, norm='l2')

	# Extract the categorical variables and perform one hot encoding.
	reader_categorical = z.select_dtypes(include=[object])
	reader_label_transformed = reader_categorical.apply(le.fit_transform)

	x = reader_numerical
	y = dataset.loc[1:,['genre']].values

	#use datascaler from sklearn library which scales the date by subtracting the mean
	pd.DataFrame(data = x, columns = features).head()

	#two component PCA, using sklearn library
	pca = PCA(n_components=components)
	principalComponents = pca.fit_transform(x)
	target_names = []

	for i in range(components):
		target_names.append('principal component ' + str(i+1))

	principaldataset = pd.DataFrame(data = principalComponents, columns = target_names)

	finaldataset = pd.concat([principaldataset, dataset.genre.shift(-1)], axis = 1)
	finaldataset = finaldataset.drop(finaldataset.index[len(finaldataset)-1])

	return finaldataset, principalComponents
