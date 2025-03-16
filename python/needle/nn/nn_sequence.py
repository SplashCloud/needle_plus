"""The module.
"""
from typing import List
from needle.core.tensor import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

def setItem(t: Tensor, axis: int, index: int, ele: Tensor) -> Tensor:
    tt = ops.split(t, axis=axis)
    lst = list(tt.tuple())
    lst[index] = ele
    return ops.stack(lst, axis=axis)

def getItem(t: Tensor, axis: int, index: int) -> Tensor:
    tt = ops.split(t, axis=axis)
    return tt[index]

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return init.ones(*x.shape, device=x.device, dtype=x.dtype, requires_grad=True) / (1 + ops.exp(-x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        k = 1/self.hidden_size
        self.W_ih = Parameter(init.rand(
            *(self.input_size, self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        ))
        self.W_hh = Parameter(init.rand(
            *(self.hidden_size, self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        ))
        self.bias_ih = Parameter(init.rand(
            *(self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )) if self.bias else None
        self.bias_hh = Parameter(init.rand(
            *(self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )) if self.bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(
                *(X.shape[0], self.hidden_size,),
                device=self.device,
                dtype=self.dtype,
                requires_grad=True
            )
        h_next = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            h_next += self.bias_ih.broadcast_to(h_next.shape) + self.bias_hh.broadcast_to(h_next.shape)
        if self.nonlinearity == 'tanh':
            return ops.tanh(h_next)
        if self.nonlinearity == 'relu':
            return ops.relu(h_next)
        raise NotImplementedError("Only support tanh and relu nonlinearity")
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        self.rnn_cells = []
        for i in range(self.num_layers):
            input_size0 = self.input_size if i == 0 else self.hidden_size
            self.rnn_cells.append(RNNCell(
                input_size=input_size0,
                hidden_size=self.hidden_size,
                bias=self.bias,
                nonlinearity=self.nonlinearity,
                device=self.device,
                dtype=self.dtype
            ))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            h0 = init.zeros(
                *(self.num_layers, X.shape[1], self.hidden_size,),
                device=self.device,
                dtype=self.dtype,
                requires_grad=True
            )
        O = init.zeros(
            *(X.shape[0], X.shape[1], self.hidden_size,),
            device=self.device,
            dtype=self.dtype,
        )
        H = init.zeros(
            *h0.shape,
            device=self.device,
            dtype=self.dtype
        )
        for i in range(self.num_layers):
            h = getItem(h0, axis=0, index=i)
            for t in range(X.shape[0]):
                if i == 0:
                    x = getItem(X, axis=0, index=t)
                else:
                    x = getItem(O, axis=0, index=t)
                h = self.rnn_cells[i](x, h)
                O = setItem(O, axis=0, index=t, ele=h)
            H = setItem(H, axis=0, index=i, ele=h)
        return O, H
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        k = 1/self.hidden_size
        self.W_ih = Parameter(init.rand(
            *(self.input_size, 4*self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        ))
        self.W_hh = Parameter(init.rand(
            *(self.hidden_size, 4*self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        ))
        self.bias_ih = Parameter(init.rand(
            *(4*self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )) if self.bias else None
        self.bias_hh = Parameter(init.rand(
            *(4*self.hidden_size,),
            low=-math.sqrt(k),
            high=math.sqrt(k),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )) if self.bias else None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = (
                init.zeros(
                    *(X.shape[0], self.hidden_size,),
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=True
                ),
                init.zeros(
                    *(X.shape[0], self.hidden_size,),
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=True
                )
            )
        sigmoid = Sigmoid()
        h0, c0 = h
        result = X @ self.W_ih + h0 @ self.W_hh # (bs, 4*hidden_size)
        if self.bias:
            result += self.bias_ih.broadcast_to(result.shape) + self.bias_hh.broadcast_to(result.shape)
        result = ops.reshape(result, shape=(X.shape[0], 4, self.hidden_size))
        tt = ops.split(result, axis=1)
        i, f, g, o = sigmoid(tt[0]), sigmoid(tt[1]), ops.tanh(tt[2]), sigmoid(tt[3])
        c1 = f * c0 + i * g
        h1 = o * ops.tanh(c1)
        return (h1, c1)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.lstm_cells = []
        for i in range(self.num_layers):
            input_size0 = self.input_size if i == 0 else self.hidden_size
            self.lstm_cells.append(LSTMCell(
                input_size=input_size0,
                hidden_size=self.hidden_size,
                bias=self.bias,
                device=self.device,
                dtype=self.dtype
            ))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = (
                init.zeros(
                    *(self.num_layers, X.shape[1], self.hidden_size,),
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=True
                ),
                init.zeros(
                    *(self.num_layers, X.shape[1], self.hidden_size,),
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=True
                )
            )
        O = init.zeros(
            *(X.shape[0], X.shape[1], self.hidden_size),
            device=self.device,
            dtype=self.dtype
        )
        hn = init.zeros(
            *(self.num_layers, X.shape[1], self.hidden_size,),
            device=self.device,
            dtype=self.dtype
        )
        cn = init.zeros(
            *(self.num_layers, X.shape[1], self.hidden_size,),
            device=self.device,
            dtype=self.dtype
        )
        for i in range(self.num_layers):
            hs = getItem(h[0], axis=0, index=i)
            chs = getItem(h[1], axis=0, index=i)
            for t in range(X.shape[0]):
                if i == 0:
                    x = getItem(X, axis=0, index=t)
                else:
                    x = getItem(O, axis=0, index=t)
                hs, chs = self.lstm_cells[i](x, (hs, chs))
                O = setItem(O, axis=0, index=t, ele=hs)
            hn = setItem(hn, axis=0, index=i, ele=hs)
            cn = setItem(cn, axis=0, index=i, ele=chs)
        return (O, (hn, cn))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(
            *(self.num_embeddings, self.embedding_dim,),
            mean=0,
            std=1,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        ))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype, requires_grad=True)
        result = one_hot.reshape((-1, self.num_embeddings)) @ self.weight
        return result.reshape((x.shape[0], x.shape[1], self.embedding_dim))
        ### END YOUR SOLUTION