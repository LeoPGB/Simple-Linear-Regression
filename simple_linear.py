#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression

df = pd.read_csv("csvfile.csv")

plot.xlabel('Area')
plot.ylabel('Price')
plot.scatter(df.Area, df.Price, color='red', marker='+')

reg = LinearRegression()
reg.fit(df[['Area']], df.Price)
print(reg.predict([[50]]))

print(reg.coef_)
print(reg.intercept_)
# %%