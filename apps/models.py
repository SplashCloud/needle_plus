import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
from needle import ops
np.random.seed(0)

class ConvBN(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, device=None, dtype="float32"):
        self.model = nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype),
            nn.BatchNorm2d(dim=out_channels, device=device, dtype=dtype),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.model(x)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBN(16, 32, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                    ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
                )
            ),
            ConvBN(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBN(64, 128, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                    ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_mode = seq_model
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype
        
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.output_size,
            embedding_dim=self.embedding_size,
            device=self.device,
            dtype=self.dtype
        )
        if self.seq_mode == 'rnn':
            self.model = nn.RNN(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                device=self.device,
                dtype=self.dtype
            )
        elif self.seq_mode == 'lstm':
            self.model = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                device=self.device,
                dtype=self.dtype
            )
        elif self.seq_mode == 'transformer':
            self.model = nn.Transformer(
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                device=self.device,
                dtype=self.dtype
            )
        else:
            raise NotImplementedError("Only support RNN, LSTM and Transformer.")
        self.classifier_in_features = self.hidden_size
        if self.seq_mode == 'transformer':
            self.classifier_in_features = self.embedding_size
        self.classifier = nn.Linear(
            in_features=self.classifier_in_features,
            out_features=self.output_size,
            device=self.device,
            dtype=self.dtype
        )
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        embedding = self.embedding_layer(x)
        output, hs = self.model(embedding, h)
        prob = self.classifier(ops.reshape(output, shape=(-1, self.classifier_in_features)))
        return prob, hs
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
