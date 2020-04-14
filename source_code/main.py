#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import sklearn
import matplotlib as mp
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import RidgeCV
import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')


# ********************Cleaning the data ********************
duplicate_rows = data[data.duplicated()]
print (len(duplicate_rows))
clean_data = data.drop_duplicates(keep=False,inplace=False)
c = clean_data
# Change the data types
c['range_Valence'] = c.range_Valence.astype(float)
c['number_of_elements'] = c.number_of_elements.astype(float)
c['range_atomic_radius'] = c.range_atomic_radius .astype(float)

c.info()


# ********************Creating The Test Set ********************

# In[4]:


train_set, test_set = train_test_split(c, test_size=0.2)
# In[6]:

# ********************Visualising the Data ********************
# thermal conductivity, atomic radius, valence, electron affinity, atomic mass
# Figure 1 in report. Plot showing the temperatures
top = ["critical_temp","wtd_std_ThermalConductivity","range_atomic_radius", "wtd_entropy_atomic_mass", "range_fie", "number_of_elements" ]
train_set[top].hist(bins=50, figsize=(20,15))
plt.show()



# ********************Looking for correlations********************
corr = train_set.corr()
top_features = ["critical_temp","wtd_std_ThermalConductivity", "range_atomic_radius", "wtd_entropy_atomic_mass",
               "range_fie", "number_of_elements", "entropy_Valence" ]

bot_feat = corr['critical_temp'].sort_values(ascending = True).head(10)
bot = bot_feat.index.values
most_corr = train_set[bot]

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
correlation_heatmap(most_corr)

top_features =["critical_temp","wtd_std_ThermalConductivity", "range_atomic_radius", "wtd_entropy_atomic_mass", 
               "range_fie", "number_of_elements", "entropy_Valence" ]


# ********************Selecting and Training a Model********************
#Prepare the data for the model
X_train = train_set.iloc[:,0:80].to_numpy()
y_train = train_set.iloc[:,81].to_numpy()

def run_linreg_train(X,y):
    linreg = LinearRegression()
    linreg.fit(X,y)
    y_hat = linreg.predict(X)
    mse = mean_squared_error(y,y_hat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_hat)
    print("Linear Regression Performance:\nRMSE = {0}\nR2 = {1}".format(rmse,r2))
    return y_hat

def run_forest_train(X,y):
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X,y)
    y_hat = regr.predict(X)
    lin_mse = mean_squared_error(y,y_hat)
    lin_rmse = np.sqrt(lin_mse)
    r2 = r2_score(y, y_hat)
    print("Random Forest Performance:\nRMSE = {0}\nR2 = {1}".format(lin_rmse,r2))
    return y_hat
    
y_hat_lin = run_linreg_train(X_train,y_train)
y_hat_forest = run_forest_train(X_train,y_train)

def plotModel(y, y_hat,c):
    # Create a dictionary to pass to matplotlib
    # These settings make the plots readable on slides, feel free to change
    # This is an easy way to set many parameters at once
    fontsize = "30";
    params = {'figure.autolayout':True,
              'legend.fontsize': fontsize,
              'figure.figsize': (12, 8),
             'axes.labelsize': fontsize,
             'axes.titlesize': fontsize,
             'xtick.labelsize':fontsize,
             'ytick.labelsize':fontsize}
    plt.rcParams.update(params)
    
    # Create a new figure and an axes objects for the subplot
    # We only have one plot here, but it's helpful to be consistent
    fig, ax = plt.subplots()
    
    # Draw a scatter plot of the first column of x vs second column.
    ax.scatter(y,y_hat, color = c)
    ax.set_xlabel("Observed Critical Temperature")
    ax.set_ylabel("Predicted Critical Temperature")
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    
    m, b = np.polyfit(y, y_hat, 1)
    ax.plot(y, m*y + b, color='red')

lin_color = 'blue'
for_color = 'green'
plotModel(y_train,y_hat_lin,lin_color)
plotModel(y_train,y_hat_forest,for_color)


linreg = LinearRegression()
forreg = RandomForestRegressor(max_depth=2, random_state=0)
# ********************Selecting and Training a Model********************
def cross_validate(reg):
    scores = cross_val_score(reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# ***************************************Fine-Tuning the model ***************************************

# ********************Grid Search********************
                                                                                                                    

                                                                                                                    
# Implementing Grid Search to find the best learning rate and momentum                                              
param_grid_forrest = [ {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8], 'max_depth': [2, 5, 10,15]},]      


def grid_search_forest(param_grid):                                                                              
    """                                                                                                   
    Runs grid search and exports data into a csv                                                          
    :return:                                                                                              
    """""                                                                                                 
    grid_search_fit = GridSearchCV(estimator=forreg,                                                         
                                   param_grid=param_grid,                                                 
                                   scoring='neg_mean_squared_error', return_train_score=True,                                                   
                                   cv=5,                                                                 
                                   n_jobs=1)                                                              
    grid_search_fit = grid_search_fit.fit(X_train, y_train)                                               
    grid_csv = pd.DataFrame(grid_search_fit.cv_results_).to_csv('grid_search_results.csv')     
    
# ********************Regularised Models********************

ridge_reg = Ridge(alpha=1, solver="cholesky")
param_grid_ridge= [{'alpha': [1, 0.5, 0.2, 0.3, 0.7, 2, 0.8, 5, 10] , 
                           'solver': ["cholesky", "sgd", "svd","lsqr","sparse_cg","sag", "saga"]}]
param_grid_lasso= [ {'alpha': [1, 0.5, 0.2, 0.3, 0.7, 2, 0.8, 5, 10]}]

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)



#lasso
lasso_reg = Lasso(alpha=0.1)


def grid_search_ridge(param_grid):                                                                              
    """                                                                                                   
    Runs grid search and exports data into a csv                                                          
    :return:                                                                                              
    """""                                                                                                 
    grid_search_fit = GridSearchCV(estimator=ridge_reg,                                                         
                                   param_grid=param_grid,                                                 
                                   scoring='neg_mean_squared_error', return_train_score=True,                                                   
                                   cv=5,                                                                 
                                   n_jobs=1)                                                              
    grid_search_fit = grid_search_fit.fit(scaled_X_train, y_train)                                               
    grid_csv = pd.DataFrame(grid_search_fit.cv_results_).to_csv('grid_search_results_ridge.csv')
    
def grid_search_lasso(param_grid):                                                                              
    """                                                                                                   
    Runs grid search and exports data into a csv                                                          
    :return:                                                                                              
    """""                                                                                                 
    grid_search_fit = GridSearchCV(estimator=lasso_reg,                                                         
                                   param_grid=param_grid,                                                 
                                   scoring='neg_mean_squared_error', return_train_score=True,                                                   
                                   cv=5,                                                                 
                                   n_jobs=1)                                                              
    grid_search_fit = grid_search_fit.fit(X_train, y_train)                                               
    grid_csv = pd.DataFrame(grid_search_fit.cv_results_).to_csv('grid_search_results_lasso.csv')
       
    
grid_search_ridge(param_grid_ridge)
grid_search_lasso(param_grid_lasso)


# ********************Stacking********************



estimators = [
    ('Random Forest', RandomForestRegressor(max_features = 14, max_depth  = 10, n_estimators =10, random_state=0)),
    ('Linear Regression', LinearRegression()),
    ('Ridge', Ridge(alpha=0.2, solver="cholesky"))
]
stacking_regressor = StackingRegressor(
    estimators=estimators, final_estimator=RidgeCV()
)


# In[29]:


def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)


# In[44]:


X = X_train
y = y_train

fig, axs = plt.subplots(2, 2, figsize=(20, 19))
axs = np.ravel(axs)

for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor',
                                               stacking_regressor)]):
    start_time = time.time()
    score = cross_validate(est, X, y,
                           scoring=['r2', 'neg_mean_squared_error'],
                           n_jobs=-1, verbose=0)
    elapsed_time = time.time() - start_time

    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)
    joblib.dump(est, "my_model.pkl")
    plot_regression_results(
        ax, y, y_pred,
        name,
        (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$RME={:.2f} \pm {:.2f}$')
        .format(np.mean(score['test_r2']),
                np.std(score['test_r2']),
                np.sqrt(-np.mean(score['test_neg_mean_squared_error']) ),
                np.sqrt(np.std(score['test_neg_mean_squared_error']))),
        elapsed_time)


plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


# ********************Evaluating on the test set********************



final_model= joblib.load("final_model.pkl")
X_test = test_set.iloc[:,0:80].to_numpy()
y_test = test_set.iloc[:,81].to_numpy()

def run_final_model(X,y, model):
    model.fit(X_train,y_train)
    y_hat = model.predict(X)
    mse = mean_squared_error(y,y_hat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_hat)
    print("Final Model Performance:\nRMSE = {0}\nR2 = {1}".format(rmse,r2))
    return y_hat

color = "black"
y_hat_final_model = run_final_model(X_test,y_test, final_model)
plotModel(y_test,y_hat_final_model,color)

