# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#read train file

df = pd.read_csv('./Datas/homeprices.csv')
area = df.drop('price', axis=1)
price = df.price

#fit the datas

reg = linear_model.LinearRegression()
reg.fit(area, price)

#import test file
test_data = pd.read_csv('./Datas/areas.csv')

#guess the price
#and save guessed price at guessed_price.csv
price_prediction = reg.predict(test_data)
test_data['AI_price'] = price_prediction
test_data.to_csv('./Datas/guessed_price.csv', index=False)

#show train datas with green color and show test datas with red color and
# show LinearRegression Line with blue color

plt.scatter(df.area, df.price,color='green')
plt.plot(area, reg.predict(area))
plt.scatter(test_data.area, test_data.AI_price, color='red')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(us$)')
plt.show()
