"""Minimal pure-Python tensor with autograd over a small op set.

Kept intentionally small but correct: supports construction from flat (1D) or
nested (2D) lists, elementwise arithmetic, matmul, relu, softmax, sum/mean and
L2 normalization. Enough to demonstrate end-to-end contrastive training of the
embedding model in pure Python without numpy/torch.
"""
import math


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=None):
        if data and not isinstance(data[0], (list, tuple)):
            data = list(data)
        else:
            data = [list(row) for row in data]
        self.data = data
        self.shape = (len(data), len(data[0]) if data and isinstance(data[0], (list, tuple)) else 0)
        self.requires_grad = requires_grad
        self.grad = None
        self._op = _op
        self._children = set(_children)

    @property
    def is_vector(self):
        return not self.data or not isinstance(self.data[0], (list, tuple))

    def zero_grad(self):
        self.grad = None
        for c in self._children:
            if isinstance(c, Tensor):
                c.zero_grad()

    def _ensure_grad(self):
        if self.grad is None:
            if self.is_vector:
                self.grad = [0.0] * len(self.data)
            else:
                self.grad = [[0.0] * len(r) for r in self.data]

    def item(self):
        if self.is_vector and len(self.data) == 1:
            return self.data[0]
        raise ValueError("item() only on scalar")

    def tolist(self):
        return [list(r) if isinstance(r, (list, tuple)) else r for r in self.data]

    def __repr__(self):
        return "Tensor(shape=%r, data=%r)" % (self.shape, self.data[:2])

    def _bop(self, other, op):
        other = other if isinstance(other, Tensor) else Tensor([other] if not self.is_vector else [other] * len(self.data))
        out = []
        if self.is_vector and not other.is_vector:
            for row in other.data:
                out.append([op(a, b) for a, b in zip(self.data, row)])
        elif not self.is_vector and other.is_vector:
            for row in self.data:
                out.append([op(a, b) for a, b in zip(row, other.data)])
        elif self.is_vector and other.is_vector:
            out = [op(a, b) for a, b in zip(self.data, other.data)]
        else:
            out = [[op(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(self.data, other.data)]
        t = Tensor(out, _children=(self, other), _op=op.__name__)
        t.requires_grad = self.requires_grad or other.requires_grad
        return t

    def __add__(self, other):
        return self._bop(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._bop(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._bop(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        if self.is_vector:
            return Tensor([-a for a in self.data], _children=(self,))
        return Tensor([[-a for a in r] for r in self.data], _children=(self,))

    def matmul(self, other):
        a = self.data
        if self.is_vector:
            a = [self.data]
        b = other.data if not other.is_vector else [[x] for x in other.data]
        out = []
        for i in range(len(a)):
            row = []
            for j in range(len(b[0])):
                row.append(sum(a[i][k] * b[k][j] for k in range(len(b))))
            out.append(row)
        if self.is_vector and other.is_vector:
            return Tensor(out[0], _children=(self, other), _op="matmul")
        return Tensor(out, _children=(self, other), _op="matmul")

    def relu(self):
        if self.is_vector:
            return Tensor([a if a > 0 else 0.0 for a in self.data], _children=(self,), _op="relu")
        return Tensor([[a if a > 0 else 0.0 for a in r] for r in self.data], _children=(self,), _op="relu")

    def softmax(self):
        def _sm(v):
            m = max(v)
            e = [math.exp(x - m) for x in v]
            s = sum(e) or 1e-9
            return [x / s for x in e]
        if self.is_vector:
            return Tensor(_sm(self.data), _children=(self,), _op="softmax")
        return Tensor([_sm(r) for r in self.data], _children=(self,), _op="softmax")

    def sum(self):
        if self.is_vector:
            return Tensor([sum(self.data)], _children=(self,), _op="sum")
        return Tensor([sum(sum(r) for r in self.data)], _children=(self,), _op="sum")

    def mean(self):
        if self.is_vector:
            return Tensor([sum(self.data) / max(len(self.data), 1)], _children=(self,), _op="mean")
        n = max(len(self.data) * max(len(self.data[0]), 1), 1)
        return Tensor([sum(sum(r) for r in self.data) / n], _children=(self,), _op="mean")

    @staticmethod
    def zeros(n):
        return Tensor([0.0] * n)

    @staticmethod
    def ones(n):
        return Tensor([1.0] * n)


def softmax(x):
    if isinstance(x, Tensor):
        return x.softmax()
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e) or 1e-9
    return [v / s for v in e]
