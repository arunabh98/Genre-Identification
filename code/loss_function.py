# Import libraries.
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def sample_loss(params):
	# This is the name of the training file.
	training_filename = "../datasets/msd.csv"

	# Read the csv file.
	reader = pd.read_csv(training_filename)

	# Drop the ID of song as it has no information.
	reader.drop(reader.columns[[1, 2, 3]], axis=1, inplace=True)

	# Some columns have string values in numerical columns.
	# So cleaning that.
	for i in range(1, reader.shape[1]):
		reader = reader[pd.to_numeric(reader[reader.columns[i]], errors='coerce').notnull()]

	# Convert string values to float.
	reader = reader.convert_objects(convert_numeric=True)

	# LabelEncoder - assigns number 1-n
	le = preprocessing.LabelEncoder()

	# Extract the numerical features and perform normalization.
	reader_numerical = reader.select_dtypes(exclude=[object])
	reader_numerical = np.matrix(reader_numerical).astype("double")
	reader_numerical = preprocessing.normalize(reader_numerical, norm='l2')

	# Extract the categorical variables and perform one hot encoding.
	reader_categorical = reader.select_dtypes(include=[object])
	reader_label_transformed = reader_categorical.apply(le.fit_transform)

	# Put the categorical and numerical features together.
	reader = np.concatenate((reader_label_transformed, reader_numerical), axis=1)

	# Initialize the decision tree classifier.
	# clf = tree.DecisionTreeClassifier()
	clf = RandomForestClassifier(n_estimators=params[0], min_samples_split=params[1], min_samples_leaf=params[2])

	return cross_val_score(estimator=clf, X=reader[:, 1:], y=reader[:, 0]).mean()