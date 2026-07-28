"""
Take Performer as T2T Transformer, code borrowd from T2T
"""
import math
import torch
import torch.nn as nn
import numpy as np

class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):#dim: The channel dimension of the input token (such as the dimension of each token in Transformer). in_dim: The dimension of each "head" or the underlying dimension to be mapped to. head_cnt: Number of attention heads (default 1 here). dp1 / dp2: two dropout ratios
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here #Define the total dimension of attention (the dimension of multiple heads). When head_cnt=1, emb=in_dim.
        self.kqv = nn.Linear(dim, 3 * self.emb)                                #If head=1 is set, it is equivalent to self-attention. The token dimension is 320 dimensions. First ×3, and then divided into 3 parts. Each part has 320 dimensions. These three parts are used to calculate Q, K, and V respectively.
        self.dp = nn.Dropout(dp1)                                              #dropout in the attention block (put it on the QK^T weight or output, depending on how to use it in forward).
        self.proj = nn.Linear(self.emb, self.emb)                              #Linear projection after attention output (out_proj in standard Transformer), shape (B, N, emb) -> (B, N, emb).
        self.head_cnt = head_cnt                                               #records the number of heads. Note: There is no explicit implementation of multi-head splitting/merging in the current code (no steps to view/reshape into (B, N, head, dim_head) are seen), indicating that there is a high probability that the "single head" is currently running.
        self.norm1 = nn.LayerNorm(dim)                                         #The first LayerNorm, usually used for normalization before/after Attn (PreNorm or PostNorm, depending on forward).
        self.norm2 = nn.LayerNorm(self.emb)                                    #The second LayerNorm, usually used for normalization before/after MLP. Because the input/output channel of MLP is emb.
        self.epsilon = 1e-8  # for stable in division #Small constant to prevent the denominator from being 0 when performing division and normalization
        self.drop_path = nn.Identity()                                         #is used to put Stochastic Depth (residual paths are randomly discarded). Now it's Identity, which is equivalent to not turning it on. If you are going to support DropPath, it is common to have a drop_path_prob and then replace it with a custom DropPath module.

        self.mlp = nn.Sequential(                                              #Transformer’s Feedforward Network (FFN).
            nn.Linear(self.emb, 1 * self.emb),                                 #Make a linear projection of the vector ∞RD_in of each token, and the input and output dimensions are both emb. Generally speaking nn.Linear(self.emb, hidden)
            nn.GELU(),                                                         #is a nonlinear activation function. Its significance/value mainly lies in: allowing the network to introduce smoother and more probabilistic nonlinearity when doing channel mapping, thereby taking into account both optimization stability and expressive ability.
            nn.Linear(1 * self.emb, self.emb),                                 #performs a linear mapping again, which is to compress the activated "high-dimensional features" back to the original dimensions and use them to add to the residuals.
            nn.Dropout(dp2),                                                   #This step is not "must make the linearity more accurate", but is used to prevent overfitting.
        )

        self.m = int(self.emb * kernel_ratio)#defines the number of random features m (kernel approximation dimension in Performer). kernel_ratio=0.5 → m ≈ 0.5*emb. The larger m → the more accurate the approximation, but the calculation is slower and takes up more video memory.
        self.w = torch.randn(self.m, self.emb)#samples a random matrix W ∈ ℝ^{m×emb}, which is used to throw Q, K into the random feature space (φ(Q)=f(QW^T,...)).
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)#uses orthogonal initialization (orthogonal) to make the row vectors of W orthogonal in pairs, and the values are more stable. Multiply by √m for scale adjustment (common implementations will have 1/√m normalization in the mapping; multiply √m first here, and then offset it with division/normalization steps to keep the variance at an appropriate level). requires_grad=False: do not train this matrix (random features are fixed)

    def prm_exp(self, x):                                                      #Relationship with MHSA: It is an approximate replacement (linear attention) of the "calculation attention" within MHSA, which is part of the attention branch. Standard MHSA: softmax(QKᵀ)V (O(N²)), Performer-MHSA: approximated with φ and linearly calculated according to the above formula (O(Nm))
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)               #Standard MHSA: For each query, similarity (N×N) needs to be done with all keys, which is expensive.
#Performer: First cast Q/K into an m-dimensional random base and perform positive value → only perform two multiplications in this m-dimensional, such as rewriting "global pairwise similarity" into "first compress to m-dimensional, and then aggregate in m-dimensional".
    def attn(self, x):                                                              # [8, 16384, 256]
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer, use v as skip connection#The final residuals are added

        return y

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))#The final residual sum
        return x
