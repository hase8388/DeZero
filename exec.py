#%%
from dezero import Variable
import numpy as np

#%%
# step24
def sphere(x, y):
    return x ** 2 + y ** 2


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
x.grad


# %%


# %%
