# Import libraries.
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree

# This is the name of the training file.
training_filename = "msd_genre_dataset.csv"

# Use all columns except the last 3.
cols_to_use = columns[:len(columns)-3]

# Read the csv file.
reader = pd.read_csv(training_filename, usecols=cols_to_use)

# Drop the ID of song as it has no information.
reader.drop(reader.columns[[1]], axis=1, inplace=True)

# Extract the genre and convert it to panda form.
genre = reader[reader.columns[0]].values.reshape(reader.shape[0], 1)
genre = pd.DataFrame(data=genre)

# LabelEncoder - assigns number 1-n
le = preprocessing.LabelEncoder()
# One hot encoder.
enc = preprocessing.OneHotEncoder()

# Extract the categorical variables and perform one hot encoding.
reader_categorical = reader.select_dtypes(include=[object])
reader_label_transformed = reader_categorical.apply(le.fit_transform)
enc.fit(reader_label_transformed)
onehotlabels = enc.transform(reader_label_transformed).toarray()

# Extract the numerical features and perform normalization.
reader_numerical = reader.select_dtypes(exclude=[object])
reader_numerical = np.matrix(reader_numerical).astype("double")
reader_numerical = preprocessing.normalize(reader_numerical, norm='l2')

# Put the categorical and numerical features together.
reader = np.concatenate((onehotlabels, reader_numerical), axis=1)

# Give labels to the genres.
genre = genre.apply(le.fit_transform)

# Split the dataset into training and test.
reader_training = reader[0:192, :]
reader_test = reader[192:, :]
genre_training = genre[0:192]
genre_test = genre[192:]

# Initialize the decision tree classifier.
clf = tree.DecisionTreeClassifier()

# Train on the model.
clf = clf.fit(reader, genre)

# Predict on the test examples.
pred = clf.predict(reader_test)

# Calculate the error and check for the accuracy.
total_error = np.sum(pred - genre_test.values.reshape(1, genre_test.shape[0]), axis=1)
accuracy = 1 - total_error[0]/genre_test.shape[0]