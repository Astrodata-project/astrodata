from astrodata.ml.metrics.BaseMetric import BaseMetric
import numpy as np

class Accuracy(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return (y_true == y_pred).mean()
    
    def get_name(self):
        return "accuracy"


class F1Score(BaseMetric):
    def __init__(self, average="macro"):
        self.average = average
        
    def __call__(self, y_true, y_pred, **kwargs):
        average = self.average
        labels = kwargs.get("labels", None)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        tp = np.zeros(len(labels))
        fp = np.zeros(len(labels))
        fn = np.zeros(len(labels))
        for i, label in enumerate(labels):
            tp[i] = np.sum((y_true == label) & (y_pred == label))
            fp[i] = np.sum((y_true != label) & (y_pred == label))
            fn[i] = np.sum((y_true == label) & (y_pred != label))
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1_per_class = np.divide(2 * precision * recall, precision + recall,
                                 out=np.zeros_like(precision), where=(precision + recall) != 0)
        if average == "macro":
            return np.mean(f1_per_class)
        elif average == "micro":
            total_tp = np.sum(tp)
            total_fp = np.sum(fp)
            total_fn = np.sum(fn)
            if (2 * total_tp + total_fp + total_fn) == 0:
                return 0.0
            precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0.0
            recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0.0
            if (precision_micro + recall_micro) == 0:
                return 0.0
            return 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        elif average == "weighted":
            support = np.array([np.sum(y_true == label) for label in labels])
            if np.sum(support) == 0:
                return 0.0
            return np.average(f1_per_class, weights=support)
        else:
            return f1_per_class  # Return per-class F1 scores

    def get_name(self):
        return "f1"
    

class ConfusionMatrix(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # Optionally accept "labels" in kwargs
        labels = kwargs.get("labels", None)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        n_labels = len(labels)
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        cm = np.zeros((n_labels, n_labels), dtype=int)
        for t, p in zip(y_true, y_pred):
            i = label_to_index[t]
            j = label_to_index[p]
            cm[i, j] += 1
        return cm

    def get_name(self):
        return "confusion_matrix"
    
class Precision(BaseMetric):
    def __init__(self, average="macro"):
        self.average = average
        
    def __call__(self, y_true, y_pred, **kwargs):
        average = self.average
        labels = kwargs.get("labels", None)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        tp = np.zeros(len(labels))
        fp = np.zeros(len(labels))
        for i, label in enumerate(labels):
            tp[i] = np.sum((y_true == label) & (y_pred == label))
            fp[i] = np.sum((y_true != label) & (y_pred == label))
        precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
        if average == "macro":
            return np.mean(precision_per_class)
        elif average == "micro":
            total_tp = np.sum(tp)
            total_fp = np.sum(fp)
            if total_tp + total_fp == 0:
                return 0.0
            return total_tp / (total_tp + total_fp)
        elif average == "weighted":
            support = np.array([np.sum(y_true == label) for label in labels])
            if np.sum(support) == 0:
                return 0.0
            return np.average(precision_per_class, weights=support)
        else:
            return precision_per_class  # Return per-class precision

    def get_name(self):
        return "precision"

class Recall(BaseMetric):
    def __init__(self, average="macro"):
        self.average = average
        
    def __call__(self, y_true, y_pred, **kwargs):
        average = self.average
        labels = kwargs.get("labels", None)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        tp = np.zeros(len(labels))
        fn = np.zeros(len(labels))
        for i, label in enumerate(labels):
            tp[i] = np.sum((y_true == label) & (y_pred == label))
            fn[i] = np.sum((y_true == label) & (y_pred != label))
        recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
        if average == "macro":
            return np.mean(recall_per_class)
        elif average == "micro":
            total_tp = np.sum(tp)
            total_fn = np.sum(fn)
            if total_tp + total_fn == 0:
                return 0.0
            return total_tp / (total_tp + total_fn)
        elif average == "weighted":
            support = np.array([np.sum(y_true == label) for label in labels])
            if np.sum(support) == 0:
                return 0.0
            return np.average(recall_per_class, weights=support)
        else:
            return recall_per_class  # Return per-class recall

    def get_name(self):
        return "recall"