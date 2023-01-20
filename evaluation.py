import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_score_metrics(true:np.ndarray, pred:np.ndarray, exclude_self=False, lag=False):
    assert np.shape(true)==np.shape(pred)
    if np.max(true) == np.min(true): #can't compute metrics
            return None,None
    if exclude_self:
        if lag: #lag inference
            dim,lag,_ = true.shape
            self_ind = np.eye(dim)[:,np.newaxis,:] * np.ones((dim,lag,dim))
        else: #variable inference
            self_ind = np.eye(true.shape[0])
        true = true[np.logical_not(self_ind)].flatten()
        pred = pred[np.logical_not(self_ind)].flatten()

        auroc = roc_auc_score(true,pred)
        auprc = average_precision_score(true,pred)
    else:
        auroc = roc_auc_score(true.flatten(),pred.flatten())
        auprc = average_precision_score(true.flatten(),pred.flatten())
    return auroc, auprc

def compute_binary_metrics(true:np.ndarray, pred:np.ndarray, exclude_self=False, lag=False):
    assert np.shape(true)==np.shape(pred)
    if np.max(true)==np.min(true):
        return None,None,None
    if exclude_self:
        if lag: #lag inference
            dim,lag,_ = true.shape
            self_ind = np.eye(dim)[:,np.newaxis,:] * np.ones((dim,lag,dim))
        else: #variable inference
            self_ind = np.eye(true.shape[0])
        true = true[np.logical_not(self_ind)]
        pred = pred[np.logical_not(self_ind)]
    true, pred = true.flatten(), pred.flatten()
    fscore = f1_score(true, pred)
    
    fp = np.sum(np.logical_and(true==0,pred==1))
    fn = np.sum(np.logical_and(true==1,pred==0))
    pp = np.sum(pred)
    p = np.sum(true)
    fdr = fp/pp
    fnr = fn/p
    return fscore, fdr, fnr


def compute_adjusted_sensitivity(true:np.ndarray, pred:np.ndarray, exclude_self=True, lag=False):
    assert np.shape(true)==np.shape(pred)
    if np.max(true)==np.min(true):
        return None,None,None
    if exclude_self:
        if lag: #lag inference
            dim,lag,_ = true.shape
            self_ind = np.eye(dim)[:,np.newaxis,:] * np.ones((dim,lag,dim))
        else: #variable inference
            self_ind = np.eye(true.shape[0])
        true = true[np.logical_not(self_ind)].flatten()
        pred = pred[np.logical_not(self_ind)].flatten()
        
    ind = np.logical_and(true!=0,pred!=0)
    true,pred = true[ind],pred[ind]
    adj_sens = np.mean(true==pred)
    return adj_sens

