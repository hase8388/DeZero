"""
https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/
"""

import numpy as np
import weakref
import contextlib


def as_array(x):
    return np.array(x) if np.isscalar(x) else x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj

    return Variable(obj)


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):

    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.generation = 0

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # 逆伝搬を使用する際のみ、outputとinputを参照として保持する
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # 弱参照にすることで、循環参照でなくし、メモリを回収されるようにする
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return (gy, gy)


def add(x0, x1):
    return Add()(x0, x1)


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


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x0, gy * x1


def mul(x0, x1):
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    return Sub()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = (1 / x1) * gy
        gx1 = (-x0 / (x1) ** 2) * gy
        return gx0, gx1


def div(x0, x1):
    return Div()(x0, x1)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0].data
        return self.c * (x ** (self.c - 1)) * gy


def pow(x, c):
    return Pow(c)(x)


class Variable:
    # 演算子の関数を優先的に呼ぶために、大きな数に設定する
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if (data is not None) and not (isinstance(data, np.ndarray)):
            raise TypeError(f"{type(data)} is not supported.")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        self.grad = np.ones_like(self.data) if self.grad is None else self.grad

        funcs = []
        seen_set = set()

        def add_func(f):
            # funcsに同じ関数を複数入れないためにチェック
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                # generationが大きいものからpopできるように並び替え
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            # Functionのoutputsを弱参照にしたことに伴い、output()としている
            gys = [output().grad for output in f.outputs]
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
                    add_func(x.creator)

            # retain_grad=Falseならば、中間のgradはすべて削除される
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def clearngrad(self):
        self.grad = None


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = sub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = div
    Variable.__pow__ = pow
