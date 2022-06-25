import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

csv_path = "Data/housing.csv"
dataframe = pd.read_csv(csv_path)

# приносимо вибачення від всієї команди за не зовсім естетично привабливий код
# ми з пайтоном на Ви

# замінюємо нульові значення середнім по колонці

average_total_bedrooms = dataframe["total_bedrooms"].astype("float").mean(axis=0)
dataframe['total_bedrooms'].replace(np.nan, average_total_bedrooms, inplace = True)

# створюємо датафрейм з dummy-variables

dummy_ocean_proximity = pd.get_dummies(dataframe['ocean_proximity'])
dummy_ocean_proximity.rename(columns={'<1H OCEAN':'ocean_proximity_<1hour_ocean',
                                      'INLAND':'ocean_proximity_inland',
                                     'ISLAND':'ocean_proximity_island',
                                      'NEAR BAY':'ocean_proximity_near_bay',
                                      'NEAR OCEAN':'ocean_proximity_near_ocean'}, inplace=True)
dataframe = pd.concat([dataframe, dummy_ocean_proximity], axis=1)
dataframe.drop('ocean_proximity', axis=1, inplace = True)

# коригуємо типи даних
dataframe['total_rooms'] = dataframe['total_rooms'].astype(int)
dataframe['population'] = dataframe['population'].astype(int)
dataframe['households'] = dataframe['households'].astype(int)

# множинна лінійна регресія

prices = dataframe['median_house_value']
features = dataframe.drop('median_house_value', axis=1)
x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)

yhat_test_lin = lr.predict(x_test)
y_test.to_csv('linearActual.csv')
np.savetxt("linearPred.csv", yhat_test_lin, delimiter=",")
yhat_train_lin = lr.predict(x_train)
train_rmse = np.sqrt(metrics.mean_squared_error(y_train, yhat_train_lin))
test_rmse = np.sqrt(metrics.mean_squared_error(y_test, yhat_test_lin))

print("Кореневе середнє квадратичне відхилення для тренувальних даних: ", train_rmse)
print("Кореневе середнє квадратичне відхилення для тестових даних: ", test_rmse)


# поліноміальна регресія

train_RMSE_list=[]
test_RMSE_list=[]



for degree in range(1,5):

    polynomial_converter= PolynomialFeatures(degree=degree, include_bias=False)
    poly_features= polynomial_converter.fit_transform(features)


    x_train, x_test, y_train, y_test = train_test_split(poly_features, prices, test_size=0.2, random_state=1)

    polymodel=LinearRegression()
    polymodel.fit(x_train, y_train)

    y_train_pred=polymodel.predict(x_train)
    y_test_pred=polymodel.predict(x_test)

    train_RMSE=np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))

    test_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

    y_test.to_csv("poly" + str(degree) + "actual.csv")
    np.savetxt("poly" + str(degree) + "pred.csv", y_test_pred, delimiter=",")


    train_RMSE_list.append(train_RMSE)
    test_RMSE_list.append(test_RMSE)

print("Кореневе середнє квадратичне відхилення для тренувальних даних:")
print(train_RMSE_list)
print("Кореневе середнє квадратичне відхилення для тестових даних:")
print(test_RMSE_list)




