import numpy as np
from sklearn.model_selection import train_test_split

xdata = np.linspace(-1, 1, 101)
ydata = 2 * xdata + np.random.randn(*xdata.shape) * 0.33

x_train, x_test, y_train, y_test =  train_test_split(xdata, ydata, test_size=.30, random_state=42)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


