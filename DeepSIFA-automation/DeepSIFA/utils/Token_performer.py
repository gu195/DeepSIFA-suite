"""
Take Performer as T2T Transformer, code borrowd from T2T
"""
import math
import torch
import torch.nn as nn
import numpy as np

class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):#dim：输入 token 的通道维（例如 Transformer 里每个 token 的维度）。in_dim：每个“头”的维度或要映射到的基础维。head_cnt：注意力头数（这里默认 1）。dp1 / dp2：两个 dropout 比例
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here         #定义注意力总维度（多头拼起来的维度）。当 head_cnt=1 时，emb=in_dim。
        self.kqv = nn.Linear(dim, 3 * self.emb)                                #若设置head=1，相当于当头自注意力，令牌维度为320维，先×3，然后分为3份，每份维度都是320维，这三份分别去计算Q，K，V。
        self.dp = nn.Dropout(dp1)                                              #注意力块里的 dropout（放在 QK^T 权重上或输出上，具体看 forward 里怎么用）。
        self.proj = nn.Linear(self.emb, self.emb)                              #注意力输出后的线性投影（标准 Transformer 里的 out_proj），形状 (B, N, emb) -> (B, N, emb)。
        self.head_cnt = head_cnt                                               #记录头数。注意：当前代码里未见对多头拆分/合并的显式实现（没看到 view/reshape 成 (B, N, head, dim_head) 的步骤），说明目前大概率按“单头”在跑
        self.norm1 = nn.LayerNorm(dim)                                         #第一个 LayerNorm，通常用于 Attn 前/后 的归一化（PreNorm 或 PostNorm，取决于 forward）。
        self.norm2 = nn.LayerNorm(self.emb)                                    #第二个 LayerNorm，通常用于 MLP 前/后 的归一化。因为 MLP 的输入/输出通道是 emb。
        self.epsilon = 1e-8  # for stable in division                          #小常数，做除法归一化时防止分母为 0
        self.drop_path = nn.Identity()                                         #用来放 Stochastic Depth（残差路径随机丢弃）。现在是 Identity，相当于不开启。如果你准备支持 DropPath，常见会有一个率 drop_path_prob，然后用自定义 DropPath 模块替换它。

        self.mlp = nn.Sequential(                                              #Transformer 的前馈网络（FFN）。
            nn.Linear(self.emb, 1 * self.emb),                                 #对每个 token 的向量 𝑥RD_in做一次线性投影，输入和输出维度都是emb。通常来说nn.Linear(self.emb, hidden)
            nn.GELU(),                                                         #一个非线性的激活函数，它的意义/价值主要在于：让网络在做通道映射时引入平滑、概率感更强的非线性，从而兼顾优化稳定性和表达能力。
            nn.Linear(1 * self.emb, self.emb),                                 #再次进行一次线性映射，是把经过激活后的“高维特征”再压回原维度，用来和残差相加
            nn.Dropout(dp2),                                                   #这一步不是“必须让线性更准”，而是防过拟合用的
        )

        self.m = int(self.emb * kernel_ratio)#定义随机特征个数 m（Performer 里的核近似维度）。kernel_ratio=0.5 → m ≈ 0.5*emb。m 越大 → 近似越准，但算得更慢更占显存。
        self.w = torch.randn(self.m, self.emb)#采样一个随机矩阵 W ∈ ℝ^{m×emb}，用于把 Q,K 投到随机特征空间（φ(Q)=f(QW^T,...)）。
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)#用 正交初始化（orthogonal）让 W 的行向量两两正交，数值更稳，乘上 √m 进行尺度调整（常见实现里会在映射里有 1/√m 的归一化；这里先乘 √m，后续多半会配合除法/归一步骤抵消，保持方差在合适量级），requires_grad=False：不训练这个矩阵（随机特征固定）

    def prm_exp(self, x):                                                      #和 MHSA 的关系：它是 MHSA 内部“算注意力”的一种近似替代（线性注意力），属于注意力分支的一部分，标准 MHSA：softmax(QKᵀ)V（O(N²)），Performer-MHSA：用 φ 近似后按上式线性算（O(Nm)）
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

        return torch.exp(wtx - xd) / math.sqrt(self.m)               #标准 MHSA：对每个 query，要和所有 key 做相似度（N×N），成本高。
#Performer：先把 Q/K 投到 m 维随机基底并做正值化 → 只在这个 m 维里做两次乘法，像是把“全局两两相似度”改写成“先压缩到 m 维，再在 m 维里聚合”。
    def attn(self, x):                                                              # [8, 16384, 256]
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer, use v as skip connection#最后残差相加

        return y

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))#最后残差相加
        return x
