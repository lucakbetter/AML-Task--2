from sklearn.model_selection import KFold


def CV_score(x_init, y_init, model_func, score, n_splits=3):
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
	total_score = 0
	
	for train_idx, test_idx in kf.split(x_init):
		
		x = x_init[train_idx]
		y = y_init[train_idx]
		x_test = x_init[test_idx]
		y_test = y_init[test_idx]
		
		y_pred = model_func(x, y, x_test)
		total_score += score(y_test, y_pred)
	return total_score/n_splits