from __future__ import annotations

from typing import Callable

import numpy as np


class Tensor:
    def __init__(
        self,
        data: np.ndarray | list | tuple | float | int,
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
        _op: str = '',
    ) -> None:
        self.data = np.array(data, dtype=float)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad

        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    @staticmethod
    def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if grad.shape == shape:
            return grad

        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, size in enumerate(shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad

    def __add__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            out_grad = out.grad
            if out_grad is None:
                return
            if self.requires_grad:
                grad_self = Tensor._sum_to_shape(out_grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._sum_to_shape(out_grad, other.data.shape)
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other: Tensor) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor) -> Tensor:
        return self + (other * Tensor(-1))

    def __rsub__(self, other: Tensor) -> Tensor:
        return other + (self * Tensor(-1))

    def __mul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            out_grad = out.grad
            if out_grad is None:
                return
            if self.requires_grad:
                grad_self = Tensor._sum_to_shape(out_grad * other.data, self.data.shape)
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._sum_to_shape(out_grad * self.data, other.data.shape)
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other
        out._backward = _backward
        return out

    def __rmul__(self, other: Tensor) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='/')

        def _backward():
            out_grad = out.grad
            if out_grad is None:
                return
            if self.requires_grad:
                grad_self = Tensor._sum_to_shape(out_grad / other.data, self.data.shape)
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._sum_to_shape(-out_grad * self.data / (other.data ** 2), other.data.shape)
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other
        out._backward = _backward
        return out

    def __rtruediv__(self, other: Tensor) -> Tensor:
        return other / self

    def __neg__(self) -> Tensor:
        return self * Tensor(-1)

    def __pow__(self, power: Tensor) -> Tensor:
        out = Tensor(self.data ** power.data, requires_grad=self.requires_grad or power.requires_grad,
                     _children=(self, power), _op='**')

        def _backward():
            out_grad = out.grad
            if out_grad is None:
                return
            if self.requires_grad:
                grad_self = Tensor._sum_to_shape(
                    out_grad * (power.data * (self.data ** (power.data - 1))),
                    self.data.shape
                )
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if power.requires_grad:
                grad_power = Tensor._sum_to_shape(
                    out_grad * (out.data * np.log(self.data + 1e-20)),
                    power.data.shape
                )
                if power.grad is None:
                    power.grad = grad_power.copy()
                else:
                    power.grad += grad_power
        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            out_grad = out.grad
            if out_grad is None:
                return
            if self.requires_grad:
                grad_self = out_grad @ other.data.T
                if self.grad is None:
                    self.grad = grad_self.copy()
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out_grad
                if other.grad is None:
                    other.grad = grad_other.copy()
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                     _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if grad is None:
                    return
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

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
            denom = np.prod([self.data.shape[ax] for ax in axes])
        return self.sum(axis=axis, keepdims=keepdims) * Tensor(1.0 / denom)

    def T(self) -> Tensor:
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')

        def _backward():
            if self.requires_grad:
                out_grad = out.grad
                if out_grad is None:
                    return
                grad = out_grad.T
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                out_grad = out.grad
                if out_grad is None:
                    return
                grad = out_grad * out.data
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                out_grad = out.grad
                if out_grad is None:
                    return
                grad = out_grad / self.data
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        out = Tensor(1 / (1 + np.exp(-self.data)), requires_grad=self.requires_grad, _children=(self,), _op='sigmoid')

        def _backward():
            if self.requires_grad:
                out_grad = out.grad
                if out_grad is None:
                    return
                grad = out_grad * out.data * (1 - out.data)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                out_grad = out.grad
                if out_grad is None:
                    return
                grad = out_grad * (1 - out.data ** 2)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                out_grad = out.grad
                if out_grad is None:
                    return
                grad = out_grad * (self.data > 0).astype(float)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def softmax(self) -> Tensor:
        exp_z = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_data = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                out_grad = out.grad
                if out_grad is None:
                    return
                dot = np.sum(out_grad * out.data, axis=-1, keepdims=True)
                grad = out.data * (out_grad - dot)
                if self.grad is None:
                    self.grad = grad.copy()
                else:
                    self.grad += grad
        out._backward = _backward
        return out

    def backward(self, grad: np.ndarray | None = None) -> None:
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

    def numpy(self) -> np.ndarray:
        return self.data

    def zero_grad(self) -> None:
        self.grad = None
