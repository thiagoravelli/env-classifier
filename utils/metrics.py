from sklearn.metrics import accuracy_score, f1_score

def accuracy(preds, labels):
    return accuracy_score(labels, preds)

def macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro")
