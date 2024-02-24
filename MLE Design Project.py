"""
Maximum Likelihood Estimator Design Project

Author: Ray Lopez
Last Updated: 2/23/24
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize


#Defining our distribution
mu = 2
sigma = 3

x = np.linspace(-10, 13)
y = norm.pdf(x, loc = mu, scale = sigma)

#Definining a x1 and x2 for visualization 
x1 = 5
x2 = -2

#Plotting this distribution and x1 and x2 points.
fig, ax = plt.subplots()
fig.set_size_inches(12,9)
fig.set_dpi(300)
ax = plt.plot(x, y, 'k')
plt.title("Baseline Distribution")
plt.grid(True)
plt.plot(x1, 0, 'bo')
plt.plot(x2, 0, 'bo')
plt.show()
plt.clf()

#Sanity check of our pdf values with what is being visualized.
l1 = norm.pdf(x1, loc=mu, scale = sigma)
l2 = norm.pdf(x2, loc=mu, scale = sigma)
#display(l1)
#display(l2)

#Generating Our Test Data
np.random.seed(25) #Lucky #25 Baby
N = 1000
x_values = np.random.normal(loc = mu, scale = sigma, size = (N,))

#Visualizing this data
fig, ax = plt.subplots()
fig.set_size_inches(12,9)
fig.set_dpi(300)
ax = plt.plot(x_values, 'k')
plt.title("Generated Data Distribution")
plt.show()
plt.clf()

#Checking to see if our generated data is reasonably close to our set mu and sigma parameters.
display("Mean of generated data: " + str(np.mean(x_values)))
display("Standard Deviation of generated data: " + str(np.std(x_values)))
#Close enough

#Defining our log likelihood function to pass to scipy minimize later
def log_likelihood(p, x):
    mu = p[0]
    sigma = p[1]
    
    l = np.sum(np.log(norm.pdf(x, loc = mu, scale = sigma)))
    
    return -l

#Defining our constraints function
def constraint(p):
    sigma = p[1]
    
    return sigma

#Defining our constraints
cons = {'type':'ineq', 'fun':constraint}
p0 =[0, 1]
display(minimize(log_likelihood, p0, args=(x_values,), constraints=cons))