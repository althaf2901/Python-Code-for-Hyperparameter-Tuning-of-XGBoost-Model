get_ipython().system('pip install xgboost')
get_ipython().system('pip install scikit-optimize')
get_ipython().system('pip install tensorflow')
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.sampler import Sobol
import pandas as pd
import numpy as np
df_train= pd.read_csv('C://Users//CCES-14//Desktop//alm//T//axial//cc_xgboost//T_axial_training.csv')
df_train = df_train[['alpha', 'beta', 'gamma', 'tau','scf_cc']]
df_val = pd.read_csv('C://Users//CCES-14//Desktop//alm//T//axial//cc_xgboost//T_axial_validation.csv')
df_val = df_val[['alpha', 'beta', 'gamma', 'tau','scf_cc']]
df_test = pd.read_csv('C://Users//CCES-14//Desktop//alm//T//axial//cc_xgboost//T_axial_testing.csv')
df_test = df_test[['alpha', 'beta', 'gamma', 'tau','scf_cc']]
#data preprocessing
#if output is NaN
nan_index = []
for i in range(len(df_train['scf_cc'])):
  if np.isnan(df_train['scf_cc'][i])==True:
    nan_index.append(i)
df_train = df_train.drop(index = nan_index)
df_train = df_train.reset_index(drop=True)
#scf_cc<0.4
scf_index = []
for i in range(len(df_train['scf_cc'])):
  if df_train['scf_cc'][i]<0:
    scf_index.append(i)
df_train = df_train.drop(index = scf_index)
df_train = df_train.reset_index(drop=True)
nan_index = []
for i in range(len(df_val['scf_cc'])):
  if np.isnan(df_val['scf_cc'][i])==True:
    nan_index.append(i)
df_val = df_val.drop(index = nan_index)
df_val = df_val.reset_index(drop=True)
#scf_cc<0.4
scf_index = []
for i in range(len(df_val['scf_cc'])):
  if df_val['scf_cc'][i]<0:
    scf_index.append(i)
df_val = df_val.drop(index = scf_index)
df_val = df_val.reset_index(drop=True)
nan_index = []
for i in range(len(df_test['scf_cc'])):
  if np.isnan(df_test['scf_cc'][i])==True:
    nan_index.append(i)
df_test = df_test.drop(index = nan_index)
df_test = df_test.reset_index(drop=True)
#scf_cc<0.4
scf_index = []
for i in range(len(df_test['scf_cc'])):
  if df_test['scf_cc'][i]<0:
    scf_index.append(i)
df_test = df_test.drop(index = scf_index)
df_test = df_test.reset_index(drop=True)
#merging both the pandas dataframes
from sklearn.preprocessing import MinMaxScaler
# Stack the DataFrames on top of each other
df1 = pd.concat([df_train, df_val,df_test], ignore_index=True)
dfi = pd.DataFrame()
dfi = dfi.append(df1.iloc[:, 0:4])
dfo = pd.DataFrame()
dfo = dfo.append(df1.iloc[:, 4:5])
from sklearn.preprocessing import MinMaxScaler
# scale = StandardScaler()
scale = MinMaxScaler()
scaleddfi = scale.fit_transform(dfi)
scaleddfo = scale.fit_transform(dfo)
#dividing into training and testing for scf_cc
#loading the dataset
from sklearn.model_selection import train_test_split
X_train = scaleddfi[0:len(df_train), :] #.to_numpy()
y_train = scaleddfo[:,0][0:len(df_train)].reshape(len(df_train), 1) #for scf_bs
# y = scaleddfo[:,0].reshape(X.shape[0], 1)#.to_numpy() #transpose
X_val = scaleddfi[len(df_train):len(df_train)+len(df_val), :]
y_val = scaleddfo[:,0][len(df_train):len(df_train)+len(df_val)].reshape(len(df_val), 1)
X_test = scaleddfi[len(df_train)+len(df_val):, :]
y_test = scaleddfo[:,0][len(df_train)+len(df_val):].reshape(len(df_test), 1)
# Initialize the Sobol sampler
sampler = Sobol()
# Define the search space for the hyperparameters
search_space = [
    Integer(1, 1000, name='n_estimators'),
    Integer(1, 30, name='max_depth'),
    Real(0.001, 0.5, name='learning_rate'),
    Integer(1, 30, name='min_child_weight'),
    Categorical(['gbtree','gblinear'], name='booster'), 
]
# Initialize the optimizer with the search space and sampler
optimizer = Optimizer(dimensions=search_space, random_state=123, base_estimator="GP", acq_func="EI", acq_optimizer="lbfgs")
# Define the objective function to optimize
@use_named_args(search_space)
def objective(**params):
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        min_child_weight=params['min_child_weight'],
        booster=params['booster'],
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return rmse
# Generate the initial points using Sobol sequence
n_points = 500
initial_points = sampler.generate(search_space, n_points)
# Create a list to store the trial points, optimization points, and best hyperparameters
points = []
# Evaluate the initial points and tell the optimizer the corresponding function values
for point in initial_points:
    value = objective(point)
    optimizer.tell(point, value)
    points.append([point, value, 'trial'])
# Continue the optimization process
n_iterations = 100
for i in range(n_iterations):
    next_point = optimizer.ask()
    value = objective(next_point)
    optimizer.tell(next_point, value)
    points.append([next_point, value, 'optimization'])
columns = ['n_estimators','max_depth', 'learning_rate', 'min_child_weight','booster','val_rmse','point_type']
# Create an empty DataFrame with the desired column names
df = pd.DataFrame()
temp_df1 = pd.DataFrame()
temp_df2 = pd.DataFrame()
# Iterate through the list and append each row to the DataFrame
for row in points:
    temp_df1 = temp_df1.append([row[0]])
for row in points:
    temp_df2 = temp_df2.append([row[1:]])
df = pd.concat([temp_df1, temp_df2], axis=1)
df.columns = columns
df = df[columns]
df = df.reset_index()
df.loc[df['val_rmse'].idxmin(), 'point_type'] = 'best'
df.to_csv('C://Users//CCES-14//Desktop//alm//T//axial//cc_xgboost//axial_cc_dp.csv')
# Save the history of the model
best_params = df.loc[df['val_rmse'].idxmin()]
# Create and train the final Random Forest model with the best hyperparameters
final_model = XGBRegressor(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    learning_rate=(best_params['learning_rate']),
    min_child_weight=int(best_params['min_child_weight']),
    booster=(best_params['booster']),
    random_state=0
)
final_model.fit(X_train, y_train)
# Evaluate the final model on the test data
y_pred_test = final_model.predict(X_test)
test_rmse = mean_squared_error(y_test_all, y_pred_test, squared=False)
print("Best hyperparameters: ", best_params)
print("Corresponding validation RMSE: ", best_params['val_rmse'])
print("RMSE on test data: ", test_rmse)
y_predict_test = final_model.predict(X_test)
y_predict_rescale_all = []
min = dfo['scf_cc'].min() #for scf_bs
max = dfo['scf_cc'].max() #for scf_bs
for i in range(0,y_predict_test.size,1):
  y_predict_rescale_all.append((y_predict_test[i]*(max-min))+min)
y_predict_rescale_all=np.array(y_predict_rescale_all)
q = y_predict_rescale_all
df_test['nn_scf_cc'] = q
df_test.to_csv('axial_cc_xgboost__database_test.csv')
y_predict_val = final_model.predict(X_val)
y_predict_rescale_val = []
min = dfo['scf_cc'].min() #for scf_bs
max = dfo['scf_cc'].max() #for scf_bs
for i in range(0,y_predict_val.size,1):
  y_predict_rescale_val.append((y_predict_val[i]*(max-min))+min)
y_predict_rescale_val=np.array(y_predict_rescale_val)
r = y_predict_rescale_val
df_val['nn_scf_cc'] = r
df_val.to_csv('axial_cc_xgboost__database_val.csv')
y_predict_train = final_model.predict(X_train)
y_predict_rescale_train = []
min = dfo['scf_cc'].min() #for scf_bs
max = dfo['scf_cc'].max() #for scf_bs
for i in range(0,y_predict_train.size,1):
  y_predict_rescale_train.append((y_predict_train[i]*(max-min))+min)
y_predict_rescale_train=np.array(y_predict_rescale_val)
s = y_predict_rescale_val
df_val['nn_scf_cc'] = s
df_train.to_csv('axial_cc_xgboost__database_train.csv')
