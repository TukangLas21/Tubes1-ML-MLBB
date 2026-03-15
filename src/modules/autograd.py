import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = None
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (other * Tensor(-1))

    def __rsub__(self, other):
        return other + (self * -1)

    def __mul__(self, other):
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad * self.data
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                grad_self = out.grad / other.data
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = -out.grad * self.data / (other.data ** 2)
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return other / self

    def __neg__(self):
        return self * Tensor(-1)

    def __pow__(self, power):
        out = Tensor(self.data ** power.data, requires_grad=self.requires_grad or power.requires_grad,
                     _children=(self, power), _op='**')

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * (power.data * (self.data ** (power.data - 1)))
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if power.requires_grad:
                grad_power = out.grad * (out.data * np.log(self.data + 1e-20))
                if power.grad is None:
                    power.grad = grad_power.copy()
                else:
                    power.grad += grad_power
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                     _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is None:
                    grad = np.broadcast_to(grad, self.data.shape)
                else:
                    grad = np.expand_dims(grad, axis=axis) if not keepdims else grad
                    grad = np.broadcast_to(grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        denom = np.prod(self.data.shape) / (np.prod(self.data.shape if axis is None else np.array(self.data.shape)[axis]))
        return self.sum(axis=axis, keepdims=keepdims) * Tensor(1.0 / denom)

    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                grad = out.grad * out.data
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                grad = out.grad / self.data
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), requires_grad=self.requires_grad, _children=(self,), _op='sigmoid')

        def _backward():
            if self.requires_grad:
                grad = out.grad * out.data * (1 - out.data)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                grad = out.grad * (1 - out.data ** 2)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0).astype(float)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def softmax(self):
        exp_z = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_data = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                for i in range(self.data.shape[0]):
                    y = out.data[i]
                    dy = out.grad[i]
                    # Jacobian-vector product
                    grad[i] = dy - np.sum(dy * y) * y
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for node in reversed(topo):
            node._backward()

    def numpy(self):
        return self.data

    def zero_grad(self):
        self.grad = None
