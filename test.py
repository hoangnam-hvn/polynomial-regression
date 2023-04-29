from nonlinear_regression import *  # import the NLR class from the nonlinear_regression module
import numpy as np  # import NumPy library
import matplotlib.pyplot as plt  # import Matplotlib library

# define the range of x values with step size
start, step = -3, 0.1
x = np.arange(start, -start+step, step)
X = x.reshape(-1, 1)  # reshape x to a column vector and assign it to X

# generate two sets of y values using the x values and some random noise
y1 = x**4 - 5*x**2 + 1 + np.random.uniform(-3, 3, size=x.shape)
y2 = 2*x**3 - 3*x + 4 + np.random.uniform(-3, 3, size=x.shape)

# create a 1x3 figure and three subplots
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133, projection='3d')

# plot the first set of data points in the first subplot
ax1.scatter(x, y1)
ax1.set_title('Ex1')

# plot the second set of data points in the second subplot
ax2.scatter(x, y2)
ax2.set_title('Ex2')

# perform nonlinear regression on the two sets of data using the NLR class and plot the regression line in each subplot
y_pred1 = NLR(4).fit(X, y1).predict(X)
ax1.plot(x, y_pred1, 'r')

y_pred2 = NLR(3).fit(X, y2).predict(X)
ax2.plot(x, y_pred2, 'r')

# generate a set of random 3D points and plot them in the third subplot
n_samples = 300
x = np.random.randint(-50, 50, size=(n_samples, 1))
x = np.sort(x, axis=0)
y = np.random.randint(0, 50, size=(n_samples, 1))
z = x**3 + 2*y**3 + np.random.uniform(-30**3, 30**3, size=x.shape)
ax3.scatter(x, y, z)

# perform nonlinear regression on the 3D points using the NLR class
X = np.concatenate([x, y], axis=1)
nlr = NLR(5).fit(X, z)

# create a mesh grid for the x and y values and generate z values using the nonlinear regression model
x_min, x_max = x.min() - 1, x.max() + 1
y_min, y_max = y.min() - 1, y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))
zz = nlr.predict(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)

# plot the surface of the predicted z values in the third subplot
ax3.plot_surface(xx, yy, zz, alpha=0.5, color='r')

# show the figure
plt.show()
