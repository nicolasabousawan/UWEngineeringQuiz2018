
# coding: utf-8

# In[1]:


import pandas
from sklearn import metrics
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold

dataset = pandas.read_csv('data.csv')
#print(dataset.program)

# # prepare datasets to be fed into the naive bayes model
# #predict attend class given extra hours and grade
CV =  dataset.program.reshape((len(dataset.program), 1))
classes = []
for i in CV:
	if i not in classes:
		classes.append(i[0])

for i in dataset:
	if (i!='program'):
		number = LabelEncoder()
		dataset[i] = number.fit_transform(dataset[i].astype('str'))
#print (dataset)

data = (dataset.ix[:,'field_interest':'working_with'].values).reshape((len(dataset.program), 9))

enc = OneHotEncoder()
enc.fit(data)
#print (enc.n_values_)
data = enc.transform(data).toarray()



class_var = LabelEncoder()
dataset['program'] = class_var.fit_transform(dataset['program'].astype('str'))
CV2 = dataset.program.reshape((len(dataset.program), 1))
enc.fit(CV2)
CV2 = enc.transform(CV2).toarray()

# Create model object
MB = MultinomialNB()

# # Train the model using the training sets
MB.fit(data, CV.ravel())


#print("Number of samples encountered for each (class, feature) during fitting:\n",MB.feature_count_.shape, MB.feature_count_)
# print("Smoothed empirical log probability for each class.:\n", MB.class_log_prior_)

#predict the class for each data point
predicted = MB.predict(data)
# print("Predictions:\n",MB.array([predicted]).T)


#predict the probability/likelihood of the prediction
prob_of_pred = MB.predict_proba(data)
#print("Probability of each class for the prediction: \n",prob_of_pred)

matrix = [[ (1 if max(prob_of_pred[j])==prob_of_pred[j][i] else 0) for i in range(13) ]for j in range(len(prob_of_pred))]
for i in range(len(prob_of_pred)):
	top = 0
	idxtop = 0
	second = 0
	idxsecond = 0
	third = 0
	idxthird = 0
	idxfourth = 0
	fourth = 0
	for j in range(13):
		if prob_of_pred[i][j] > top:
			idxfourth = idxthird
			idxthird = idxsecond
			third = second
			second = top
			idxsecond = idxtop
			top = prob_of_pred[i][j]
			idxtop = j
		elif prob_of_pred[i][j] > second:
			second = prob_of_pred[i][j]
			idxsecond = j
		# elif prob_of_pred[i][j] > third:
		# 	third = prob_of_pred[i][j]
		# 	idxthird = j
		# elif prob_of_pred[i][j] > fourth:
		# 	fourth = prob_of_pred[i][j]
		# 	idxfourth = j
	matrix[i][idxsecond] = 1
	# matrix[i][idxthird] = 1
	# matrix[i][idxfourth] = 1  
    
print (matrix[0], "matrix")

print("The confusion matrix:\n", metrics.confusion_matrix(CV, predicted, classes))

by_prog = []
for i, j in enumerate(metrics.confusion_matrix(CV, predicted, classes)):
	each_class = 0
	for k in j:
		each_class += k
	by_prog.append(each_class)
print("by prog", by_prog)
print("Accuracy of the model: with 1 class only",MB.score(data,CV))

# print "feature importantance", MB.coef_
# print "feature count", MB.feature_count_
print ("class count", MB.class_count_)
correct = 0
incorrect = 0
# print CV2[0]
# print matrix[0]
# print len(CV2), len(matrix)
correct_matrix = [0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(CV2)):
	for j in range(13):
		if int(CV2[i][j]) == 1 and matrix [i][j] == 1:
			correct += 1
			correct_matrix[j] += 1
            
# Calculating 5 fold cross validation results
model = MultinomialNB()
kf = KFold(len(CV), n_folds=5, random_state=None, shuffle=True)
scores = cross_val_score(model, data, CV.ravel(), cv=kf)
print("MSE of every fold in 5 fold cross validation: 1 class", abs(scores))
print("Mean of the 5 fold cross-validation: 1 class %0.2f" % abs(scores.mean()))

print ("Accuracy of the model: with top 2 probability classes", 1.0*correct/len(CV2))
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
X = data
y = CV.ravel()
acc_percent = 0
A = [0,0,0,0,0,0,0,0,0,0,0,0,0]
B = [0,0,0,0,0,0,0,0,0,0,0,0,0]
# print A
str_arr = []

for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	MB2 = MultinomialNB()
	# # Train the model using the training sets
	MB2.fit(X_train, y_train)	
	# print X_train[0], "sample x training data"
	# print y_train[0], "sample y training data"
	# print len(X_train), "length of training"
	predicted = MB2.predict(X_test)
	#predict the probability/likelihood of the prediction
	prob_of_pred = MB2.predict_proba(X_test)
	matrix = [[ (1 if max(prob_of_pred[j])==prob_of_pred[j][i] else 0) for i in range(13) ]for j in range(len(prob_of_pred))]
	print (matrix[0])
	for j in matrix:
		for i in range(13):
			if j[i] == 1:
				B[i] += 1
	for i in range(len(prob_of_pred)):
		top = 0
		idxtop = 0
		second = 0
		idxsecond = 0
		third = 0
		idxthird = 0
		idxfourth = 0
		fourth = 0
		for j in range(13):
			if prob_of_pred[i][j] > top:
				idxfourth = idxthird
				idxthird = idxsecond
				third = second
				second = top
				idxsecond = idxtop
				top = prob_of_pred[i][j]
				idxtop = j
			elif prob_of_pred[i][j] > second:
				second = prob_of_pred[i][j]
				idxsecond = j
			# elif prob_of_pred[i][j] > third:
			# 	third = prob_of_pred[i][j]
			# 	idxthird = j
			# elif prob_of_pred[i][j] > fourth:
			# 	fourth = prob_of_pred[i][j]
			# 	idxfourth = j
		matrix[i][idxsecond] = 1
		# matrix[i][idxthird] = 1
		# matrix[i][idxfourth] = 1
	correct = 0
	# print y_test
	y_test = class_var.fit_transform(y_test.astype('str'))
	y_test = y_test.reshape((len(y_test), 1))
	enc.fit(y_test)
	y_test = enc.transform(y_test).toarray()
	# print matrix
	for i in range(len(X_test)):
		for j in range(13):
			if int(y_test[i][j]) == 1 and matrix [i][j] == 1:
				correct += 1
	# print y_test[0], "y test"
	# print matrix[0], "matrix"
	# print("Number of Predictions:\n",len(predicted))
	str_arr.append(1.0*correct/len(predicted))
	acc_percent += (1.0*correct/len(predicted))
	for j in matrix:
		for i in range(13):
			if j[i] == 1:
				A[i] += 1
	print (matrix[0], "matrix")
    
    
print ("array of stratified", str_arr)
print ("the average of 5-fold is with custom NB 2 classes", acc_percent/5)
print (B, "1 class and the programs recommended by this strategy")
print (A, "2 class and the programs recommended by this strategy")
print ("correct matrix", correct_matrix)
print ("number of actual", by_prog)

