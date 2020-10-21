import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression, f_regression, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from load_data import dataframes, nanInfo, countValues
from outlier_detection import isolation_forest, local_outlier
from feature_selection import kBest
from cross_validation import CV_score

crossval = True

def preprocess(x, y, x_test):
	#	OUTLIER DETECTION
	#x_train, y_train = isolation_forest(x_train, y_train)
	x, y = local_outlier(x, y)
	
	#	SCALING
	scaler = StandardScaler().fit(x)
	x = scaler.transform(x)
	x_test = scaler.transform(x_test)
	
	#	FEATURE SELECTION
	x, x_test = kBest(x, y, x_test, f_classif, 50)
	
	return x, y, x_test
	
	
def model(x, y, x_test):
	clf = LogisticRegression(random_state=0).fit(x, y)
	y_pred = clf.predict(x_test)
	return y_pred
	

def main():
	#	DATAFRAMES
	x, y, x_test = dataframes()
	x, y, x_test = preprocess(x, y, x_test)
		
	if crossval:
		#	CROSS-VALIDATION
		score = CV_score(x, y, model, balanced_accuracy_score)	
		print(score)
	else:
		#	Y-TEST
		countValues(y, 'initial y')
		y_pred = model(x, y, x_test)
		countValues(y_pred, 'predicted y')
		
		y_test = np.concatenate((
			np.arange(y_pred.size).reshape(-1, 1),
			y_pred.reshape(-1, 1)), axis=1)
		np.savetxt(fname='y_test.csv', header='id,y', delimiter=',', X=y_test,
				fmt=['%d', '%i'], comments='')
	

if __name__ == "__main__":
	main()