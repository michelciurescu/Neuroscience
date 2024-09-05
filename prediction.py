import pickle
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import statistics
from multiprocessing import Pool
import itertools

# Load data from pickle files
with open('FCs.pkl', 'rb') as f:
    datafc = pickle.load(f)

with open('MINDs.CT_MC_MD_ICVF.pkl', 'rb') as f:
    datamind = pickle.load(f)

valid_indices = []

# Determine the valid indices in valid_indices
for n in range(len(datamind)):
    valid_count = 0
    for i in range(360):
        if len(datamind[n][i][~np.isnan(datamind[n][i])]) == 359 and len(datafc[n][i][~np.isnan(datafc[n][i])]) == 359:
            valid_count += 1
    if valid_count == 360:
        valid_indices.append(n)

#predictions = []
medians = []

def predi(i):
    x = []
    y = []
    for n in valid_indices:
        x.append(datamind[n][i][~np.isnan(datamind[n][i])])
        y.append(datafc[n][i][~np.isnan(datafc[n][i])])
    
    X = np.array(x)
    y = np.array(y)
    
    kf = KFold(n_splits=10, shuffle=False)
    fold_predictions = []
    #correlations = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ridge = Ridge(alpha=1)
        multioutput_model = MultiOutputRegressor(ridge)
        multioutput_model.fit(X_train, y_train)
        y_pred = multioutput_model.predict(X_test)
        
        fold_predictions.append(y_pred.ravel())
    
    return(fold_predictions)

# Parallel processing
if __name__ == '__main__':
    num_cpus = 3
    with Pool(num_cpus) as pool:
        pool.map(predi, range(360))
        results = pool.map(predi, range(360))
    # Separate the results into predictions and medians
    with open('results_AZI.txt', 'w') as f:
     for prediction in results:
                f.write(' '.join(map(str, prediction)) + '\n')
                f.write('\n')  # Add a blank line between different indices

    print("Results saved to 'results.txt'")
    with open('results_joi.pkl', 'wb') as f:
        pickle.dump(results, f)
    

'''
with open('coi.txt', 'r') as file:
    lines = file.readlines()


# Get the last 360 lines
last_360_lines = lines[-360:]

# Print or process the last 360 lines
for line in last_360_lines:
    print(line.strip())  
'''
