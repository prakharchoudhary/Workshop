from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import clear, plot_decision_regions

class DTClassifier:
	"""
	The DTClassifier class carries functions for:
	- Loading and splitting the dataset into train and test data.
	- Displaying important details.
	- Plotting the charts.
	- Training a decision tree classifier on iris dataset.
	- Testing results on iris dataset.
	"""

	def __init__(self):
		"""
		Constructor for the Classifier object.
		"""
		self.train_X = None
		self.test_X = None
		self.train_y = None
		self.test_y = None
		self.test_size = 0.3
		self.dtree_pkl_name = 'dtree_classifier.pkl'
		if os.path.exists('./'+self.dtree_pkl_name):
			self.load_model()
		self.load_data()

	def load_data(self):
		"""
		Load the iris dataset and split it into training and testing data
		"""
		self.dataset = load_iris()
		self.train_X, self.test_X, self.train_y, self.test_y = \
		train_test_split(self.dataset.data[:, [2,3]], self.dataset.target, test_size = self.test_size, random_state=42)
		print("-----------\n| DATASET |\n-----------\ntrain_X: {}\ntrain_y: {}\ntest_X: {}\ntest_y: {}\n"\
			.format(self.train_X.shape, self.test_X.shape, self.train_y.shape, 
				self.test_y.shape))
		return True

	def plot_data(self):
		"""
		Plot the train data using matplotlib.
		"""
		X_combined = np.vstack((self.train_X, self.test_X))
		y_combined = np.hstack((self.train_y, self.test_y))
		plot_decision_regions(X_combined, y_combined, classifier=self.model, test_idx=range(105, 150))
		plt.xlabel('petal length [cm]')
		plt.ylabel('petal width [cm]')
		plt.legend(loc='upper left')
		plt.show()

	def train(self):
		"""
		Train a the classifier:
		- simply load the machine learning model
		- Remember to tune the parameters of Decision Tree, else it might overfit
		- Fit on the training set
		- Test the accuracy
		- Save the model
		"""
		self.model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=3)
		self.model.fit(self.train_X, self.train_y)
		print("The training accuracy is: {}".format(self.model.score(self.train_X, self.train_y)))
		self.save_model()

	def save_model(self):
		"""
		Save the trained classifier in pickle format
		"""
		dtree_model_pkl = open(self.dtree_pkl_name, 'wb')
		pickle.dump(self.model, dtree_model_pkl)
		dtree_model_pkl.close()

	def load_model(self):
		"""
		Load a pretrained classification model from its pickle file.
		"""
		try:
			dtree_pkl_model = open(self.dtree_pkl_name, 'rb')
			self.model = pickle.load(dtree_pkl_model)
			print("Loaded Decision tree model :: {}".format(self.model))
			return True
		except FileNotFoundError as e:
			print("File not found: Train the model first.")
			return False

	def accuracy_test(self):
		"""
		Load the saved model and test its accuracy on test set
		"""
		if self.load_model():
			print("Accuracy of Model on testing set :: " , self.model.score(self.test_X, self.test_y))


if __name__ == '__main__':

	clear()
	clf = DTClassifier()
	while True:
		print("Select option:\n1. Train DT classifier\n2. Test its accuracy\n3. Plot the graph")
		choice = input()
		if choice == '1':
			clf.train()
		elif choice == '2':
			clf.accuracy_test()
		elif choice == '3':
			clf.plot_data()
		elif choice == 'q':
			break
		else:
			print("Invalid choice! Try Again.")

