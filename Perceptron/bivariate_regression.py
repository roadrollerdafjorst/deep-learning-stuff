# %%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# %%
data = pd.read_csv(r'data\Regression\BivariateData\24.csv', names=['x', 'y', 'z'])
target_output = data.z
data = data.drop(columns=['z'])
# %%

X_train, X_test, y_train, y_test = train_test_split(
    data, target_output, test_size=0.30, random_state=42)
X_train.insert(loc=0, column='1', value=[1 for i in range(X_train.shape[0])])
X_test.insert(loc=0, column='1', value=[1 for i in range(X_test.shape[0])])
y_train = y_train.reset_index()
y_test = y_test.reset_index()
y_train = y_train.drop(['index'], axis=1)
y_test = y_test.drop(['index'], axis=1)
# %%

w = np.array([1.0, 1.0, 1.0])
inst_errors = []
avg_error = []


def activation_function(a):
    return a


def delta_w(eta, w, y, s, x):
    delta = eta * (y-s) * x
    return w + delta


def neuron(eta, w, x, y):
    a = np.dot(w, x)
    s = activation_function(a)
    E = ((y-s)**2)/2  # instantaneous error
    inst_errors.append(E)
    w = delta_w(eta, w, y, s, x)
    return w

# %%


# Training the model
for epoch in range(1, 21):
    eta = 1/epoch
    for i in range(X_train.shape[0]):
        w = neuron(eta, w, X_train.iloc[i], y_train.iloc[i, 0])
    avg_error.append(np.mean(inst_errors))
    inst_errors = []

# %%
plt.plot(avg_error)
plt.xlabel("Average Error")
plt.xlabel("Epochs")
plt.title("Average error v/s Epochs")
plt.grid()
plt.show()

# %%
train_mse = 0
# Classification of training data
predicted = []
for i in range(X_train.shape[0]):
    predicted.append(np.dot(X_train.iloc[i], w))

# Mean squared error
train_mse = mean_squared_error(y_train, predicted)
print("MSE (training data) = ", train_mse)

# %%

# Model output and target output for training data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_train.x, X_train.y, predicted, cmap='Greens')
ax.scatter3D(X_train.x, X_train.y, y_train.z, cmap='Greens')

plt.xlabel("Input (x)")
plt.ylabel("Input (y)")
plt.title("Model output and Target output for Training data")
plt.legend(['Model Output', 'Target Output'])
plt.show()
# %%

# Target output vs model output for training data
plt.scatter(y_train.z, predicted)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S Model output (Training)")
plt.grid()
plt.show()

# %%
test_mse = 0
# Classification of testing data
predicted = []
for i in range(X_test.shape[0]):
    predicted.append(np.dot(X_test.iloc[i], w))

# Mean squared error
test_mse = mean_squared_error(y_test, predicted)
print("MSE (testing data) = ", test_mse)

# %%
# Model output and target output for testing data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test.x, X_test.y, predicted, cmap='Greens')
ax.scatter3D(X_test.x, X_test.y, y_test.z, cmap='Greens')

plt.xlabel("Input (x)")
plt.ylabel("Input (y)")
plt.title("Model output and Target output for Testing data")
plt.legend(['Model Output', 'Target Output'])
plt.grid()
plt.show()

# %%

# Target output vs model output for testing data
plt.scatter(y_test.z, predicted)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S Model output (Testing)")
plt.show()

# %%

plt.bar(['Training', 'Testing'], [train_mse, test_mse], color='maroon', width=0.4)
plt.title("Training and Testing MSE")
plt.ylabel("MSE values")
plt.grid()
plt.show()

# %%
