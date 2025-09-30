import torch
import math
from torch import nn

# set rand seed
#torch.seed(0)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device = None, dtype: torch.dtype = None):
        super(Linear, self).__init__()
        # the assignment requires d_out * d_in
        self.W = nn.Parameter(torch.empty((out_features, in_features), device = device, dtype = dtype))
        sigma = math.sqrt(2/(in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, 0, sigma, -3 * sigma, 3 * sigma)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        in transformer model, input X is typically a 3d tensor Batch * seq_len * embdding_size
        a Linear is typically map a vector size from d_in to d_out, here d_in is the embdding_size
        as W is d_out * d_in, when perform compute with pyTorch instead of einsum, we need to transpose W first
        and to perform batch mat mul, we need to repeat W.T in first dim for Batch times.
        Then the calc is torch.bmm(X, repeated W_trans)
        update: no need call bmm, matmul will do broadcast if W has fewer dimension
        '''
        # W = self.W.transpose(0,1).repeat(X.shape[0], 1, 1)
        # return torch.bmm(X, W)
        return X @ self.W.T

# uv run pytest -k test_linear
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.W = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))
        self.embedding_dim = embedding_dim
        sigma = 1
        torch.nn.init.trunc_normal_(self.W, 0, sigma, -3 * sigma, 3 * sigma)
    
    def forward(self, token_ids: torch.Tensor):
        '''
        token_ids should be Batch * Seq_len
        '''
        # device = token_ids.device
        # batch_size, seq_len = token_ids.shape[0], token_ids.shape[1]
        # ret = torch.empty((batch_size, seq_len, self.embedding_dim), device=device)
        # for batch in range(batch_size):
        #     for seq_num in range(seq_len):
        #         ret[batch][seq_num] = self.W[token_ids[batch][seq_num]]
        # return ret
        return self.W[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, X: torch.Tensor):
        # batch_size, seq_len = X.shape[0], X.shape[1]
        # device = X.device
        # in_type = X.dtype
        # ret = torch.empty((batch_size, seq_len, self.d_model), device=device)
        # for batch in range(batch_size):
        #     for seq_num in range(seq_len):
        #         data = X[batch][seq_num].to(dtype=torch.float32)
        #         RMS = ((data ** 2).sum() / self.d_model + self.eps).sqrt()
        #         data = data * self.w / RMS
        #         ret[batch][seq_num] = data
        # ret = ret.to(in_type)
        # return ret
        in_type = X.dtype
        X = X.to(dtype=torch.float32)
        #RMS = (((X ** 2).sum(-1) + self.eps)/self.d_model).sqrt().unsqueeze(-1).repeat(1,1,self.d_model)
        RMS = ((X ** 2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return (X * self.w / RMS).to(in_type)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype = None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # self.W1 = nn.Parameter(torch.ones((self.d_ff, self.d_model), device=device, dtype=dtype))
        # self.W2 = nn.Parameter(torch.ones((self.d_model, self.d_ff), device=device, dtype=dtype))
        # self.W3 = nn.Parameter(torch.ones((self.d_ff, self.d_model), device=device, dtype=dtype))
        self.W1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.W2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.W3 = Linear(self.d_model, self.d_ff, device, dtype)
    
    def forward(self, X):
        def SiLU(X):
            return X * torch.sigmoid(X)
        
        def SwiGLU(X, W1, W2, W3) -> torch.Tensor:
            return W2(SiLU(W1(X)) * (W3(X)))

        '''
        X: Batch * seq_len * d_model, we should apply SwiGLU on the last dimension
        '''
        # implement 1 loop
        # batch_size, seq_len = X.shape[0], X.shape[1]
        # device = X.device
        # ret = torch.empty((batch_size, seq_len, self.d_model), device=device)
        # for batch in range(batch_size):
        #     for seq_num in range(seq_len):
        #         data = X[batch][seq_num]
        #         ret[batch][seq_num] = SwiGLU(data, self.W1, self.W2, self.W3)
        # return ret

        # implement 2 reshape
        # orig_shape = X.shape
        # X = X.reshape(-1, self.d_model).transpose(1, 0) #after transpose d_model * n
        # X = SwiGLU(X, self.W1, self.W2, self.W3).transpose(1, 0) #after transpose n * d_model
        # return X.reshape(orig_shape)

        # implement 3 matmul, tensor with higher dim can @ or matmul tensor with lower dim, the lower will do broadcast and then batch matmul performed
        # update:rewrite with Linear Module
        X1 = self.W1(X)
        X3 = self.W3(X)  # [..., d_model] -> [..., d_ff]
        return self.W2((SiLU(X1) * X3)) #[..., d_ff] -> [..., d_model]

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_k = d_k
        index = torch.arange(0, max_seq_len).reshape(-1, 1)
        exponent = torch.arange(start=0, end=d_k, step=2) / d_k
        theta = index / torch.pow(theta, exponent)

        # we can compress R^i to a vector [R^i_1, R^i_2... R^i_d/2], when process R^i @ q, just generate [R^i_1 @ q(1, 2), R^i_2 @ q(3,4) ... R^i_d/2 @ q(d/2 -1, d/2)]
        # R^i_k = [[cos(theta^i_k), -sin], [sin, cos]], then batch mat mul can be used.
        # def build_R_elem(theta: float):
        #     return torch.Tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], device=device)
        # self.R = torch.empty((max_seq_len, d_k//2, 2, 2), device=device)
        # for i in range(max_seq_len):
        #     for k in range(d_k//2):
        #         theta_val = theta[i][k].item()
        #         self.R[i][k] = build_R_elem(theta_val)
        # self.register_buffer('RoPE_Weight', self.R)
        self.register_buffer('cos', torch.cos(theta))
        self.register_buffer('sin', torch.sin(theta))

    def forward(self, X: torch.Tensor, token_positions: torch.Tensor):
        # token_positions = token_positions.view(-1)
        # token_positions_size = token_positions.size(0)
        # orig_shape = X.shape
        # X = X.reshape(-1, self.d_k//2, 2, 1)
        # ret = torch.empty_like(X)
        # token_num = X.shape[0]
        # for i in range(token_num):
        #     token_pos = token_positions[i%token_positions_size].item()
        #     R = self.R[token_pos]
        #     token = X[i]
        #     encoded_X = torch.bmm(R, token)
        #     ret[i] = encoded_X
        # ret = ret.reshape(orig_shape)
        # return ret
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        X1 = X[..., ::2]
        X2 = X[..., 1::2]
        X_rotated = torch.stack([X1 * cos - X2 * sin, X1 * sin + X2 * cos], dim = -1)
        X_rotated = X_rotated.flatten(-2)
        return X_rotated

def softmax(X:torch.Tensor, dim:int) -> torch.Tensor:
    max = X.max(dim=dim, keepdim=True)[0]
    X = torch.exp(X - max)
    X_sum = X.sum(dim=dim, keepdim=True)
    return X/X_sum

# QK : n * d_k  V: n * d_v  use attention = QK.T, attention: n*n, attention @ V n * d_v
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
    d_model = Q.shape[-1]
    attention = Q @ K.transpose(-1, -2) / math.sqrt(d_model)
    if mask is not None:
        # we should focus on True, but masked_fill will fill when mask is True, so use ~mask
        attention = attention.masked_fill(~mask, -1e9)
    attention = softmax(attention, -1)
    return attention @ V

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope:RotaryPositionalEmbedding = None, device = None, dtype = None):
        super(MultiheadSelfAttention, self).__init__()
        # input QKV has same size n * d_in
        # each head will has a unique W_QKV_i weight(but in this assign they are same), map QKV into n * dk, total num_heads head
        # by concat all heads in row, we will get a n * dk*num_heads Tensor
        # if we concat W_QKV_i into one matrix in row, we will get the result within one step.
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.rope = rope
        # d_in = d_model
        # self.W_Q = nn.Parameter(torch.empty((self.d_model, d_model), device=device, dtype=dtype))
        # self.W_K = nn.Parameter(torch.empty((self.d_model, d_model), device=device, dtype=dtype))
        # self.W_V = nn.Parameter(torch.empty((self.d_model, d_model), device=device, dtype=dtype)) # d_k = d_v
        # self.W_O = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype)) # d_out * d_v, d_out = d_model
        # torch.nn.init.xavier_uniform_(self.W_Q)
        # torch.nn.init.xavier_uniform_(self.W_K)
        # torch.nn.init.xavier_uniform_(self.W_V)
        # torch.nn.init.xavier_uniform_(self.W_O)
        # update use Linear instead of nn.Parameter
        self.W_Q = Linear(self.d_model, self.d_model, device, dtype)
        self.W_K = Linear(self.d_model, self.d_model, device, dtype)
        self.W_V = Linear(self.d_model, self.d_model, device, dtype)
        self.W_O = Linear(self.d_model, self.d_model, device, dtype)

    def forward(self, X:torch.Tensor, token_positions:torch.Tensor = None):
        def transpose(X: torch.Tensor):
            # transform X from [..., n, d_model] -> [..., n, num_heads, d_k] ->[..., num_heads, n, d_k]
            return X.reshape(*X.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        
        def transpose_back(X: torch.Tensor):
            # transform X from [..., num_heads, n, d_k] -> [..., n, num_heads, d_k] -> [..., n, d_model]
            X = X.transpose(-2, -3)
            return X.reshape(*X.shape[:-2], self.d_model)

        def transpose_and_rope(X: torch.Tensor, token_positions:torch.Tensor):
            X = transpose(X)
            if self.rope is not None:
                X = self.rope(X, token_positions)
            return X
        Q=K=V=X
        Q = transpose_and_rope(self.W_Q(Q), token_positions)
        K = transpose_and_rope(self.W_K(K), token_positions)
        V = transpose(self.W_V(V))#[..., num_heads, n, d_k]
        num_steps = Q.shape[-2]
        device = Q.device
        mask = torch.tril(torch.ones((num_steps, num_steps), dtype=torch.bool, device=device))
        multi_head_attention = scaled_dot_product_attention(Q, K, V, mask) #[..., num_heads, n, d_v]
        multi_head_attention = transpose_back(multi_head_attention)
        return self.W_O(multi_head_attention) # n * d_out

class transformer_block(nn.Module):
    rope = None
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device = None):
        super(transformer_block, self).__init__()
        self.norm1 = RMSNorm(d_model, device=device)
        if transformer_block.rope is None:
            transformer_block.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        # pos encoding layer. maybe global variable?
        self.attention = MultiheadSelfAttention(d_model, num_heads, transformer_block.rope, device=device)
        self.norm2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)
    
    def forward(self, X: torch.Tensor):
        seq_len = X.shape[-2]
        token_positions = torch.arange(seq_len)
        Y = X + self.attention(self.norm1(X), token_positions)
        Z = Y + self.ffn(self.norm2(Y))
        return Z

class transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device = None):
        super(transformer_lm, self).__init__()
        # parameters
        self.vocab_size = vocab_size
        self.max_seq_len = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        # components
        self.embedding = Embedding(self.vocab_size, self.d_model, device=self.device) # embdding will map input token id to vector with length d_model
        blocks = []    # add num_layers transformer blocks
        for i in range(self.num_layers):
            blocks.append(transformer_block(self.d_model, self.num_heads, self.d_ff, self.max_seq_len, self.theta, device=self.device))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model, device=self.device)
        self.out_linear = Linear(d_model, vocab_size, device=self.device)
    
    def forward(self, X:torch.Tensor):
        X = self.embedding(X)
        for i in range(self.num_layers):
            X = self.blocks[i](X)
        X = self.norm(X)
        Y = self.out_linear(X)
        # as illustrated in figure1 in section 3, the returned value should be softmax, but the test will match logits......
        #return softmax(Y, -1)
        return Y

# linear = Linear(2, 3)
# X = torch.rand((5,2,2))
# Y = linear(X)
# print(Y)
# rms_norm = RMSNorm(3)
# Y = rms_norm(Y)
# print(Y)
