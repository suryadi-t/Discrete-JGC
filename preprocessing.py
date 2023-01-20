import numpy as np

def standardize_data(X:np.ndarray, y:np.ndarray):
    #if temporal, last index is time so we don't standardize it
    for j in range(X.shape[1]):
        mean,sd = np.mean(X[:,j]), np.std(X[:,j])
        X[:,j] = (X[:,j] - mean) / sd
    
    mean,sd = np.mean(y), np.std(y)
    y = (y-mean) / sd
    return X, y

def preprocess_data(data:np.ndarray, target_index:int, eta:int, contemp=False, standardize=False,
                    loss_type='poisson',loss_fn='poisson'):
    """For input features of shape (length,dim), add lags into the dim dimension.
    Arranges data as x_1(t-eta), x_2(t-eta), ..., x_1(t-eta+1), ... x_1(t-1), ..., x_i(t)
    where i != target_index."""

    X = np.hstack([data[i:-eta+i] for i in range(eta)]) #lagged vars
    
    #include contemporaneous data
    if contemp: 
        contemporaneous = np.delete(data[eta:],target_index,axis=1) #contemporaneous vars
        X = np.hstack((X,contemporaneous))
        
    y = data[eta:,target_index]
    
    if loss_type == 'hurdle':
        y = y[:,np.newaxis]
        indicator = (y>0).astype(float)
        y = np.hstack((y,indicator))
    elif loss_type == 'logistic':
        y = (y>0).astype(float) #binarize
    
    #set standardize to True only for continuous data
    if standardize: X, y = standardize_data(X, y)

    return X,y