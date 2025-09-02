# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
A - LINEAR TREND ESTIMATION

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Crypto Data Since 2015.csv')
data.head()
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
resampled_data=data['Bitcoin (USD)'].resample('Y').sum().to_frame()
resampled_data.head()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Month': 'Year'}, inplace=True)
resampled_data.head()
years=resampled_data['Date'].tolist()
bitcoin_prices=resampled_data['Bitcoin (USD)'].tolist()

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X,bitcoin_prices)]
n = len(years)
b = (n * sum(xy) - sum(bitcoin_prices) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(bitcoin_prices) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]
```
B- POLYNOMIAL TREND ESTIMATION

```
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, bitcoin_prices)]
coeff = [[len(X), sum(X), sum(x2)],
[sum(X), sum(x2), sum(x3)],
[sum(x2), sum(x3), sum(x4)]]
Y = [sum(bitcoin_prices), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

plt.figure(figsize=(10,5))
resampled_data['Bitcoin (USD)'].plot(kind='line', color='blue', marker='o', label='Bitcoin (USD)')
resampled_data['Linear Trend'].plot(kind='line', color='black', linestyle='--', label='Linear Trend')
plt.title("Bitcoin Price with Linear Trend")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show

plt.figure(figsize=(10,5))
resampled_data['Bitcoin (USD)'].plot(kind='line', color='blue', marker='o', label='Bitcoin (USD)')
resampled_data['Polynomial Trend'].plot(kind='line', color='black', marker='o', label='Polynomial Trend')
plt.title("Bitcoin Price with Polynomial Trend")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION

<img width="581" height="75" alt="image" src="https://github.com/user-attachments/assets/70c7871c-6d54-41bc-96d5-8fdb25304bc5" />

<img width="833" height="470" alt="image" src="https://github.com/user-attachments/assets/8164b960-e9d6-49f3-a0ff-d714d5c850fe" />


B- POLYNOMIAL TREND ESTIMATION

<img width="833" height="470" alt="image" src="https://github.com/user-attachments/assets/d6c01cf9-ec9a-483f-99e6-ce6353d23778" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
