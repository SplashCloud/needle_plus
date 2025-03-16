from needle import nn

class GroupedQueryAttention(nn.Module):

    def __init__(self, embed_dim, hidden_size, n_head, n_group, device):
        # base property
        self.embed_size = embed_dim
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_group = n_group
        self.device = device
        # parameter
        self.W_Q = nn.Linear(embed_dim, hidden_size, device=device)
        self.W_K = nn.Linear(embed_dim, n_group*self.head_dim, device=device)
        self.W_V = nn.Linear(embed_dim, n_group*self.head_dim, device=device)


    def forward(self, q, k=None, v=None):
        '''
        q: (bs, seq_len, embed_dim)
        '''
        if k is None:
            k = q
        if v is None:
            v = q

        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)

        