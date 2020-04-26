"""
https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/step07.py
https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/step08.py
"""

#%%
import numpy as np


#%%
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 使わない、再帰呼び出しの場合
    def backward_(self):
        f = self.creator
        if f is not None:
            x = f.input
            # 逆電版させて、inputのgradをset
            x.grad = f.backward(self.grad)
            # 再帰呼び出し
            x.backward_()

    def backward(self,):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, input):
        x = input.data
        y = self.forward(x)

        self.input = input
        output = Variable(y)
        output.set_creator(self)
        self.output = output

        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


# %%
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy

        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy

        return gx


#%%%
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

#%%
# 順伝搬
a = A(x)
b = B(a)
y = C(b)
print(y.data)

# %%
# 逆伝搬(再帰関数)
y.grad = np.array(1.0)
y.backward_()
print(x.grad)


# %%
# 逆伝搬(改良版)
y.grad = np.array(1.0)
y.backward_()
print(x.grad)


# %%
