import numpy as np
from sklearn import metrics as skmetrics

class Result:
	def __init__(self):
		self._y_true = np.array([])
		self._y_pred = np.array([])

	def update(self, y_true, y_pred):

		if np.shape(y_true) != np.shape(y_pred):
			import logging
			logging.error(f"Vectors in Result are not of the same shape: p:{np.shape(y_pred):}, l:{np.shape(y_true)}")
			raise Exception("Vectors in Result are not of the same shape")

		self._y_true = np.append(self._y_true, y_true.cpu())
		self._y_pred = np.append(self._y_pred, y_pred.cpu())

	def _threshold_preds(self, y_preds, thre_positive):
		res = (y_preds>thre_positive).astype(int)
		return res

	def confusion_matrix(self, thre_positive):
		y_pred_thre = self._threshold_preds(self._y_pred, thre_positive)
		return skmetrics.confusion_matrix(self._y_true,y_pred_thre).ravel()

	def accuracy(self,thre_positive):
		y_pred_thre = self._threshold_preds(self._y_pred, thre_positive)
		return skmetrics.accuracy_score(self._y_true,y_pred_thre)

	def specificity(self,thre_positive):
		tn, fp, fn, tp = self.confusion_matrix(thre_positive)
		return tn / (tn + fp)

	def sensitivity(self,thre_positive):
		return self.recall(thre_positive)

	def precision(self,thre_positive):
		y_pred_thre = self._threshold_preds(self._y_pred, thre_positive)
		return skmetrics.precision_score(self._y_true,y_pred_thre)

	def recall(self,thre_positive):
		y_pred_thre = self._threshold_preds(self._y_pred, thre_positive)
		return skmetrics.recall_score(self._y_true,y_pred_thre)

	def mcc(self,thre_positive):
		y_pred_thre = self._threshold_preds(self._y_pred, thre_positive)
		return skmetrics.matthews_corrcoef(self._y_true,y_pred_thre)

	def bias(self):
		return np.mean(self._y_pred) - np.mean(self._y_true)

	def roc_auc(self):
		return skmetrics.roc_auc_score(self._y_true, self._y_pred)

	def avg_precision(self):
		return skmetrics.average_precision_score(self._y_true,self._y_pred)

	def roc_curve(self):
		return skmetrics.roc_curve(self._y_true,self._y_pred)

	def pr_curve(self):
		return skmetrics.precision_recall_curve(self._y_true, self._y_pred)

	def bce(self):
		import torch
		return torch.nn.functional.binary_cross_entropy(self._y_pred,self._y_true)


if __name__ == "__main__":
	res = Result()
	res.update(np.random.randint(2,size=10), np.random.randint(2,size=10))

	print(res)
