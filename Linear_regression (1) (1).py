import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

df = pd.read_csv(r'C:/Users/Warda Ghias/Desktop/AI ENGINEERING/diamonds.csv')
df.head()
def predict_diamond_price():
    
    #to identify the most suitable independent variable for predicting the price of diamonds we first see all the numerical columns in our data set as they are our object of interest
    columns = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
    sns.pairplot(df[columns])
    df[columns].corr()
        # independent variable should be on x axis and dependent variable on y axis
        # visual methods: paiplot, more advanced heatmap ,# statistical method : correlations
        # statsmodel is used for statistical analysis it is a popular python library for statistical modeling and hypothesis testing.
        # when we use the by default statsmodel to ddo linear regression it does not automatically add a constant term (intercept) to the model. 
        # "b" so we have to explicity methion it other wise it would treat it as (0,0) origin which is not 
        # the sutiable way to predict the best fit line.To make sure your model includes both m (the slope) and b (the intercept), 
        # you’ll use the add_constant() function. This function adds a new column of ones to your x DataFrame, allowing StatsModels to calculate the intercept term.
    x = sm.add_constant(df["carat"]) # add intercept  independent variable
    y = df["price"]
    print(x)

    #ordinary least squares (OLS) regression is used to find the best-fit line (means that the distance between the predicted values and the actual values is minimized)
    model = sm.OLS(y,x)
    result = model.fit() # this step calculates m and b
    print(result.summary())   # to see the detailed results of the regression analysis
    #The best fit model equation no would be 7756 *carat + (-2256.3604)
    print(result.params)
    print(type(result.params))
    b = result.params['const']
    m = result.params['carat']
    carat = float(input("Enter the carat value of the diamond: "))
    price = m * carat + b
    return(f"The predicted price of the diamond with {carat} carat is: ${price:.2f}") 

predict_diamond_price()

def multiple_linear_regression():
    predictors = ['carat','x','y','z']
    x = sm.add_constant(df[predictors]) # add intercept  independent variable
    y = df["price"]
    model = sm.OLS(y,x).fit()
    #   Multiple Linear Regression Model to predict based on more than one independent variable as it helps to capture more complexity and improve accuracy. fro the above linear expression the accuracy is 85 percent.
    print(model.summary())
    
multiple_linear_regression
#pd.get_dummies() is a function in pandas that converts categorical variable(s) into dummy/indicator variables.example:
#df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green']}) 
#dtype - int to get numeric values for the regression model by get_dummies and drop_first = True to avoid the dummy variable trap.