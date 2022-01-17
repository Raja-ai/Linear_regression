#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
#%config InlineBackend.figure_formats = {'pdf',}
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')


# In[2]:


df = pd.read_csv('data.txt', sep = ",", names = ['population','benefice'])
df


# In[3]:


population=df.population #x
benefice=df.benefice #y
#print(population)
#print(benefice)


# In[4]:


plt.scatter('population','benefice',data=df, s=30, c='r', marker='x' )
plt.xlabel("Population of City in 10.000s")
plt.ylabel("Profit in $10.000s")
plt.xlim(4, 24)
plt.show()


# In[5]:


#matrice X
y=benefice
X=population
X


# In[6]:


theta = [0,0]
#theta.shape
theta

#Fonction de cout
# In[7]:


def computeCost(X, y, theta):
    m = len(y)
    J=0
    J= 1/(2*m) * np.sum(((theta[0]+X*theta[1]) - y)**2)
    return J


# In[8]:


computeCost(X,y,theta)

#Descente de Gradient
# In[9]:


def gradient_descent(X, y, theta, alpha, num_iters):
 # création d'un tableau de stockage pour enregistrer l'évolution du Cout
    J_history = np.zeros(num_iters)
    m = len(y)
    for i in range(0, num_iters):
        theta[0] = theta[0] - alpha * 1/m * np.sum((theta[0] + theta[1] * X)-y)
        theta[1] = theta[1] - alpha * 1/m * np.sum(((theta[0] + theta[1] * X) - y)*X)
         # mise a jour du pa
        J_history[i] = computeCost(X, y, theta) # on enregistre la valeur du
    return theta ,J_history


# enchainement 

# In[10]:


alpha=0.01
num_iters=1500
theta_final,J_historique=gradient_descent(X,y,theta,alpha,num_iters)


# In[11]:


J_historique


# In[12]:


theta_final


# In[14]:


prediction = theta[0]+X*theta[1]
plt.scatter(X,y)
plt.plot(X,prediction,c='r')


# In[15]:


plt.plot(range(num_iters), J_historique)
plt.xlabel('Iterations')
plt.ylabel('Cost_J')
plt.show()


# In[16]:


plt.scatter(population,benefice,s=30,c='red',marker='x')
plt.title('Nuage de points des populations en fonction des bénéfices ')
plt.xlabel('Population of City in 10.000s')
plt.ylabel('Profit in $10.000s')
plt.xlim(4,24)
#plt.savefig('ScatterPlot_01.png')
plt.plot(population, prediction,label="droit de regression", c='b')
plt.legend(loc = "upper left")
plt.savefig('Régression Linéaire01.png')
plt.show()


# In[17]:


print(theta_final[1])
print(theta_final[0])


# In[18]:


35.000*theta_final[0]+theta_final[1]


# In[19]:


70.000*theta_final[0]+theta_final[1]


# In[ ]:




