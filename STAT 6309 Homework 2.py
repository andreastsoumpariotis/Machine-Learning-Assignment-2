#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 23:29:30 2020

@author: andreastsoumpariotis
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot

# Question 9
auto = pd.read_csv('Auto.csv', na_values=['?'])
auto.dropna(inplace=True)
auto.reset_index(drop=True, inplace=True)

# Part a
pd.plotting.scatter_matrix(auto,figsize=(10,12))

# Part b

mpg = auto['mpg']
cylinder = auto['cylinders']
displacement = auto['displacement']
horsepower = auto['horsepower']
weight = auto['weight']
acceleration = auto['acceleration']
year = auto['year']
origin = auto['origin']

# remove the "name" column
my_cols = set(auto.columns)
my_cols.remove('name')
auto = auto[my_cols]

# mpg
np.corrcoef(mpg, cylinder)
np.corrcoef(mpg, displacement)
np.corrcoef(mpg, horsepower)
np.corrcoef(mpg, weight)
np.corrcoef(mpg, acceleration)
np.corrcoef(mpg, year)
np.corrcoef(mpg, origin)

# cylinder
np.corrcoef(cylinder, displacement)
np.corrcoef(cylinder, horsepower)
np.corrcoef(cylinder, weight)
np.corrcoef(cylinder, acceleration)
np.corrcoef(cylinder, year)
np.corrcoef(cylinder, origin)

# displacement
np.corrcoef(displacement, horsepower)
np.corrcoef(displacement, weight)
np.corrcoef(displacement, acceleration)
np.corrcoef(displacement, year)
np.corrcoef(displacement, origin)

# horsepower
np.corrcoef(horsepower, weight)
np.corrcoef(horsepower, acceleration)
np.corrcoef(horsepower, year)
np.corrcoef(horsepower, origin)

# weight
np.corrcoef(weight, acceleration)
np.corrcoef(weight, year)
np.corrcoef(weight, origin)

# acceleration
np.corrcoef(acceleration, year)
np.corrcoef(acceleration, origin)

# year
np.corrcoef(year, origin)

# Part c
from statsmodels.formula.api import ols
model = ols("mpg ~ cylinder + displacement + horsepower + weight + horsepower + acceleration + year + origin", auto).fit()
print(model.summary())

# Part d
# fitted values (need a constant term for intercept)
model_fitted_y = model.fittedvalues

# model residuals
model_residuals = model.resid

# normalized residuals
model_norm_residuals = model.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

# leverage
model_leverage = model.get_influence().hat_matrix_diag

# cook's distance
model_cooks = model.get_influence().cooks_distance[0]

# Residuals vs Fitted Plot
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'mpg', data=auto, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_residuals[i]));

# QQ Plot
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]));

# Scale-Location Plot
plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)

plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

# annotations
abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_norm_residuals_abs_sqrt[i]));
                 
# Leverage Plot
plot_lm_4 = plt.figure(4)
plot_lm_4.set_figheight(8)
plot_lm_4.set_figwidth(12)

plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
sns.regplot(model_leverage, model_norm_residuals, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_4.axes[0].set_xlim(0, 0.20)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

for i in leverage_top_3:
    plot_lm_4.axes[0].annotate(i, 
                               xy=(model_leverage[i], 
                                   model_norm_residuals[i]))

# Part e    
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
import warnings

# Suppress Warning
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
model2 = ols("mpg ~ (cylinders*displacement+displacement*weight)", auto).fit()
print(model2.summary())

# Part f

# Log
auto['log_value'] = np.log(auto['weight'])
plt.scatter(auto['log_value'], mpg)
plt.title('MPG vs. Log of Weight')
plt.ylabel('Miles Per Gallon')
plt.xlabel('Log Weight')  

# Sqrt
auto['weight_squareroot']=auto['weight']**(1/2)
plt.scatter(auto['weight_squareroot'], mpg)
plt.title('MPG vs. Sqrt of Weight')
plt.ylabel('Miles Per Gallon')
plt.xlabel('Sqrt Weight')  

# Square
auto['weight_square']=auto['weight']**(2)
plt.scatter(auto['weight_square'], mpg)
plt.title('MPG vs. Square of Weight')
plt.ylabel('Miles Per Gallon')
plt.xlabel('Square Weight') 

# Question 11
import random
from scipy import stats
random.seed(1)
x = np.random.normal(size = 100)
y = 2*x+np.random.normal(size = 100)

# Part a
df = (x,y)
dataframe = pd.DataFrame(data = df)
dataframe = np.transpose(dataframe)
model = ols("y ~ x + 0", dataframe).fit()
print(model.summary())

# Part b
model2 = ols("x ~ y + 0", dataframe).fit()
print(model2.summary())

# Part c
# For both parts a and b, I obtained the same values for the t-statistic and the p-value. 
# Essentially, the regression line equations can be written in terms of “x” and “y.”

# Part d
import math 
n = len(x)
m = np.dot(x, y, out=None)
t = math.sqrt(n - 1)*(m)/math.sqrt(sum(x**2) * sum(y**2) - (m)**2) #20.7582

# Part e
t = math.sqrt(n - 1)*(m)/math.sqrt(sum(y**2) * sum(x**2) - (m)**2) #20.7582

# Part f
# y on x
model = ols("y ~ x", dataframe).fit()
print(model.summary())
# x on y
model2 = ols("x ~ y", dataframe).fit()
print(model2.summary())

# Question 14

# Part a
random.seed(1)
x1 = np.random.uniform(size = 100)
x2 = 0.5 * x1 + np.random.normal(size = 100)/10
y = 2 + 2 * x1 + 0.3 * x2 + np.random.normal(size = 100)
# Y=2+2X1+0.3X2+ε
# Regression Coefficients: 2, 2 & 0.3 

# Part b
np.corrcoef(x1, x2)
# Scatter Plot
plt.scatter(x1, x2)
plt.ylabel('x2')
plt.xlabel('x1')

# Part c
model3 = ols("y ~ x1 + x2", dataframe).fit()
print(model3.summary())

# Part d
model4 = ols("y ~ x1", dataframe).fit()
print(model4.summary())

# Part e
model5 = ols("y ~ x2", dataframe).fit()
print(model5.summary())

# Part f
# The results obtained in (c)–(e) do not contradict each other. Both the “x1” and “x2” 
# predictors are highly correlated which means that there’s collinearity. Essentially, 
# this means that it can be difficult to determine how the predictors are related to y 
# (the response). With collinearity, we fail to reject H0 as well. 

# Part g

x1 = np.append(x1 , 0.1)
x2 = np.append(x2 , 0.8)
y = np.append(y, 6)
df = (x1, x2, y)
dataframe = pd.DataFrame(data = df)
dataframe = np.transpose(dataframe)
new_model3 = ols("y ~ x1 + x2", dataframe).fit()
print(new_model3.summary())
new_model4 = ols("y ~ x1", dataframe).fit()
print(new_model4.summary())
new_model5 = ols("y ~ x2", dataframe).fit()
print(new_model5.summary())


# Question 15
boston = pd.read_csv('Boston.csv')
boston
crim = boston['crim']
zn = boston['zn']
indus = boston['indus']
chas = boston['chas']
nox = boston['nox']
rm = boston['rm']
age = boston['age']
dis = boston['dis']
rad = boston['rad']
tax = boston['tax']
ptratio = boston['ptratio']
black = boston['black']
lstat = boston['lstat']
medv = boston['medv']

# Part a
# zn
fit_zn = ols("crim ~ zn", boston).fit()
print(fit_zn.summary())
# indus
fit_indus = ols("crim ~ indus", boston).fit()
print(fit_indus.summary())
# chas
fit_chas = ols("crim ~ chas", boston).fit()
print(fit_chas.summary())
# nox
fit_nox = ols("crim ~ nox", boston).fit()
print(fit_nox.summary())
# rm
fit_rm = ols("crim ~ rm", boston).fit()
print(fit_rm.summary())
# age
fit_age = ols("crim ~ age", boston).fit()
print(fit_age.summary())
# dis
fit_dis = ols("crim ~ dis", boston).fit()
print(fit_dis.summary())
# rad
fit_rad = ols("crim ~ rad", boston).fit()
print(fit_rad.summary())
# tax
fit_tax = ols("crim ~ tax", boston).fit()
print(fit_tax.summary())
# ptratio
fit_ptratio = ols("crim ~ ptratio", boston).fit()
print(fit_ptratio.summary())
# black
fit_black = ols("crim ~ black", boston).fit()
print(fit_black.summary())
# lstat
fit_lstat = ols("crim ~ lstat", boston).fit()
print(fit_lstat.summary())
# medv
fit_medv = ols("crim ~ medv", boston).fit()
print(fit_medv.summary())

# Part b
fit = ols("crim ~ zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat + medv", boston).fit()
print(fit.summary())

# Part c (couldn't figure out)

fit_zn[coef]

df = (modelx, modely)
dataframe = pd.DataFrame(data = df)
dataframe = np.transpose(dataframe)
dataframe = dataframe.drop(dataframe.index[0])
dataframe = pd.DataFrame(df, columns = ['x', 'y'])


boston.corr(method ='kendall') 

modelx = np.append("numeric", 0)
modely = np.append("numeric", 0)

modelx = np.append(modelx, -0.0739)
modelx = np.append(modelx,  0.5098)
modelx = np.append(modelx, -1.8928)
modelx = np.append(modelx, 31.2485)
modelx = np.append(modelx, -2.6841)
modelx = np.append(modelx, 0.1078)
modelx = np.append(modelx, -1.5509)
modelx = np.append(modelx, 0.6179)
modelx = np.append(modelx, 0.0297)
modelx = np.append(modelx, 1.1520)
modelx = np.append(modelx, -0.0363)
modelx = np.append(modelx, 0.5488)
modelx = np.append(modelx, -0.3632)

modely = np.append(modely, 0.0449)
modely = np.append(modely, -0.0639)
modely = np.append(modely, -0.7491)
modely = np.append(modely, -10.3135)
modely = np.append(modely, 0.4301)
modely = np.append(modely, 0.0015 )
modely = np.append(modely, -0.9872)
modely = np.append(modely, 0.5882)
modely = np.append(modely, -0.0038)
modely = np.append(modely, -0.2711)
modely = np.append(modely, -0.0075)
modely = np.append(modely, 0.1262)
modely = np.append(modely, -0.1989)

plt.scatter(modelx, modely) 
plt.title('Multi Reg vs. Simple Reg')
plt.ylabel('Multi Reg')
plt.xlabel('Simple Reg')
plt.ylim(-10, 0)
plt.xlim(0, 30) 


params["C[zn]"]

import statsmodels.api as sm
result = sm.OLS(crim, zn).fit()
print(result.params)

# Part d (couldn't figure out)

# zn
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=3)

fit_zn2 = np.polyfit(zn, crim, 3)

fit_zn2 = smf.ols(formula='crim ~ zn + zn + zn', data=boston).fit()
fit_zn2 = ols(formula = 'crim ~ I(zn) + I(zn**2) + I(zn**3)', data = boston).fit()
print(fit_zn2.summary())

fit_zn2 = np.polyfit(crim, zn, 3)





