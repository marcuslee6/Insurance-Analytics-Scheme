import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

# 1. read data
df = pd.read_csv('insurance.csv')

# 2. deal with string variables (One-Hot Encoding)
# sex, smoker, region change to 0 and 1
df_numeric = pd.get_dummies(df, drop_first=True)

# 3. choose variable (x)，erase target variable charges
x = df_numeric.drop(columns=['charges'])

# 4. MUST add Constant (Intercept), otherwise VIF will be incorrect
x = add_constant(x)

# 5. make sure all datatypes are float to deal with infinite/NaN
x = x.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

# 6. calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]

print(vif_data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 7. Linear Regression (predict)
y = df['charges'] 
model = sm.OLS(y, sm.add_constant(x)).fit()
print(model.summary())