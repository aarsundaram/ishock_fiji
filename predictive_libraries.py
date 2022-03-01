from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np

import numpy.polynomial.polynomial as poly

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed()

def pcanalysis(df,features,explicables,n_comps=None):

    if not (n_comps is not None): n_comps = len(features)

    # RUN PCA
    # features = [_1 for _1,_,_ in indvar]
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n_comps,svd_solver='full')
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['PC{}'.format(_+1) for _ in range(n_comps)])

    finalDf = pd.concat([principalDf, df[explicables].reset_index()], axis = 1)
    finalDf.to_csv('monte_carlo/PCA_coefficients.csv')

    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
    return finalDf

def df_to_loglog_fit(df,colX,colY,wgt=None,xlog=True,ylog=True,dummy_prediction=None):
    if xlog == True: X = np.log(df[colX].astype('float')).values.reshape(-1,1)  # values converts it into a numpy array
    else: X = df[colX].values.reshape(-1, 1)

    if ylog == True: Y = np.log(df[colY].astype('float')).values.reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
    else: Y = df[colY].values.reshape(-1, 1)
    
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X,Y,sample_weight=wgt)  # perform linear regression

    # make predictions
    if dummy_prediction is not None: Y_pred = linear_regressor.predict(np.array(dummy_prediction).reshape(-1,1))
    else: Y_pred = linear_regressor.predict(X)
        
    coef = float(linear_regressor.coef_)
    r2_score = float(linear_regressor.score(X,Y,sample_weight=wgt))
    return Y_pred,coef,r2_score

def array_to_linear_fit(X,Y,wgt=None):
    X = np.array(X).reshape(-1,1)  # values converts it into a numpy array
    Y = np.array(Y).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X,Y,sample_weight=wgt)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    r2_score = float(linear_regressor.score(X,Y,sample_weight=wgt))
    return Y_pred,r2_score

def df_to_linear_fit(df,colX,colY,wgt=None):

    X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y,sample_weight=wgt)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    coef = float(linear_regressor.coef_)
    r2_score = float(linear_regressor.score(X,Y,sample_weight=wgt))

    return Y_pred,coef,r2_score

def df_to_exponential_fit(df,colX,colY,wgt=None):

    X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

    # Y = np.log(df[colY].values.reshape(-1, 1)) # -1 means that calculate the dimension of rows, but have 1 column
    transformer = FunctionTransformer(np.log, validate=True)
    y_trans = transformer.fit_transform(Y)     

    linear_regressor = LinearRegression()  # create object for the class
    results = linear_regressor.fit(X, y_trans,sample_weight=wgt)

    linear_regressor.fit(X, y_trans,sample_weight=wgt)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    coef = float(linear_regressor.coef_)

    return Y_pred,coef

def df_to_polynomial_fit(df,colX,colY,power,wgt=None,x_new=None):

	# X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
	# Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
	X = df[colX].squeeze().T
	Y = df[colY].squeeze().T

	coefs = poly.polyfit(X,Y,power)

	if x_new is None: x_new = np.linspace(0, 40, num=100)
	ffit = poly.polyval(x_new, coefs)

	return x_new,ffit

