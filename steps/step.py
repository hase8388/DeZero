"""
https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/
"""

#%%
import numpy as np


def as_array(x):
    return np.array(x) if np.isscalar(x) else x


#%%
class Variable:
    def __init__(self, data):
        if (data is not None) and not (isinstance(data, np.ndarray)):
            raise TypeError(f"{type(data)} is not supported.")
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self,):
        self.grad = np.ones_like(self.data) if self.grad is None else self.grad

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                # すでにsetされている場合、加算する

                else:
                    x.grad = gx + x.grad

                if x.creator is not None:
                    funcs.append(x.creator)

    def clearngrad(self):
        self.grad = None


class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        self.inputs = inputs
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


#%%
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return (gy, gy)


def add(x0, x1):
    return Add()(x0, x1)


# %%
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy

        return gx


def square(x):
    return Square()(x)


#%%
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)

#%%
z = add(x, x)
z.backward()
print(x.grad)
x.clearngrad()

z2 = add(add(x, x), x)
z2.backward()
print(x.grad)
x.clearngrad()


# %%


# %%
