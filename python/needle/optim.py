"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            ## 需要注意: weight_decay是加在loss function中的
            ## 所以 weight_decay影响的是梯度的值: grad = grad + weight_decay * w
            grad = p.grad.data + self.weight_decay * p.data
            if p not in self.u.keys():
                self.u[p] = 0.0
            self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad
            p.data = p.data - self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            if p not in self.m.keys():
                self.m[p] = 0.0
                self.v[p] = 0.0
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (grad ** 2)
            hat_m = self.m[p] / (1 - self.beta1 ** self.t)
            hat_v = self.v[p] / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * (hat_m / (hat_v ** 0.5 + self.eps))
        ### END YOUR SOLUTION
