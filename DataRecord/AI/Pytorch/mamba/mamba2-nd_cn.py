"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

# 导入dataclass装饰器，用于创建数据类
from dataclasses import dataclass
# 导入NamedTuple，用于创建命名元组
from typing import NamedTuple

# 导入PyTorch库中的张量和神经网络模块
import torch
import torch.nn.functional as F
# 导入einops库中的rearrange和repeat函数，用于张量操作
from einops import rearrange, repeat
# 从PyTorch中导入Tensor类型和神经网络模块(nn)
from torch import Tensor, nn

# 定义Device变量，用于后续操作中指定计算设备
# 这里的设计是为了灵活选择运行在CPU或GPU上，取决于具体硬件配置和用户需求
Device = torch.device

@dataclass
class Mamba2Config:
    """
    Mamba2配置类，用于存储模型的各种超参数。

    参数:
        d_model (int): 模型维度 (D)
        n_layer (int, 可选): 语言模型中的Mamba-2层的数量，默认为24
        d_state (int, 可选): 状态维度 (N)，默认为128
        d_conv (int, 可选): 卷积核大小，默认为4
        expand (int, 可选): 扩展因子 (E)，默认为2
        headdim (int, 可选): 头部维度 (P)，默认为64
        chunk_size (int, 可选): 矩阵分区大小 (Q)，默认为64
        vocab_size (int): 词汇表大小
        pad_vocab_size_multiple (int): 词汇表大小的填充倍数
    """
    
    d_model: int                                            # 模型维度 (D)
    n_layer: int = 24                                       # 语言模型中的Mamba-2层数量
    d_state: int = 128                                      # 状态维度 (N)
    d_conv: int = 4                                         # 卷积核大小
    expand: int = 2                                         # 扩展因子 (E)
    headdim: int = 64                                       # 头部维度 (P)
    chunk_size: int = 64                                    # 矩阵分区大小 (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        """
        在初始化后计算内部维度和其他相关参数。
        
        计算步骤:
            1. 根据扩展因子和模型维度计算内部维度 `d_inner`。
            2. 验证内部维度是否可以被头部维度整除。
            3. 根据内部维度和头部维度计算头数量 `nheads`。
            4. 如果词汇表大小不是指定倍数的整数倍，则调整词汇表大小到下一个倍数。
        """
        # 内部维度等于扩展因子乘以模型维度
        self.d_inner = self.expand * self.d_model
        # 确保内部维度可以被头部维度整除
        assert self.d_inner % self.headdim == 0
        # 计算多头注意力的头数量
        self.nheads = self.d_inner // self.headdim
        # 如果词汇表大小不是指定倍数的整数倍，则进行调整
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)


class InferenceCache(NamedTuple):
    # 定义推理缓存的卷积状态和状态空间模型状态
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        """
        分配推理缓存的空间。

        参数:
        batch_size -- batch的大小
        args -- 包含网络架构参数的配置对象
        device -- 执行计算的设备（可选）

        返回:
        一个初始化为零的InferenceCache实例，用于存储一批样本的卷积状态和状态空间模型状态。
        """
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2(nn.Module):
    """
    Mamba2语言模型类，基于PyTorch实现。

    Mamba2模型结合了卷积和状态空间模型（SSM）的特性，用于序列数据处理，如自然语言处理任务。
    该模型通过隐藏状态传递信息，使得推理步骤具有记忆性，且推理时间随序列长度线性增长。

    参数:
        d_model: int - 模型维度D。
        n_layer: int - 语言模型中的Mamba2层数（默认24层）。
        d_state: int - 状态维度N（默认128）。
        d_conv: int - 卷积核大小（默认4）。
        expand: int - 扩展因子E（默认2）。
        headdim: int - 头维度P（默认64）。
        chunk_size: int - 矩阵分区大小Q（默认64）。
        vocab_size: int - 词汇表大小（默认50277）。
        pad_vocab_size_multiple: int - 垫片大小，用于调整词汇表大小为该值的倍数（默认16）。
    """

    def __init__(self, d_model: int,                # 模型维度 (D)
                 n_layer: int = 24,                 # 语言模型中 Mamba-2 层的数量
                 d_state: int = 128,                # 状态维度 (N)
                 d_conv: int = 4,                   # 卷积核大小
                 expand: int = 2,                   # 扩展因子 (E)
                 headdim: int = 64,                 # 头维度 (P)
                 chunk_size: int = 64,              # 矩阵分区大小 (Q)
                 vocab_size: int = 50277,
                 pad_vocab_size_multiple: int = 16
                 ):
        
        super().__init__()
        
        # 初始化配置参数
        args = Mamba2Config(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, vocab_size, pad_vocab_size_multiple)
        self.args = args

        # 输入投影层，用于将输入嵌入到更高维度空间
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False)

        # 一维卷积层，用于处理序列数据的局部相关性
        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
        )

        # 参数化时间偏移和状态转换参数
        self.dt_bias = nn.Parameter(torch.empty(args.nheads, ))
        self.A_log = nn.Parameter(torch.empty(args.nheads, ))
        self.D = nn.Parameter(torch.empty(args.nheads, ))

        # 规范化层和输出投影层
        self.norm = RMSNorm(args.d_inner, )
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, )

    def forward(self, u: Tensor, h=None):
        """
        前向传播函数，处理输入序列并更新隐藏状态。

        参数:
            u: Tensor - 输入序列，形状为(batch, seqlen, d_model)，seqlen应为chunk_size的倍数。
            h: Optional - 推理步骤的隐藏状态，如果未提供则初始化为0。

        返回:
            y: Tensor - 输出序列，形状为(batch, seqlen, d_model)。
            h: InferenceCache - 处理输入u后的更新隐藏状态。
        """
        # 如果隐藏状态提供，则执行单步推理
        if h:
            return self.step(u, h)

        # 计算注意力矩阵A和时间步长dt
        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # 卷积操作，处理序列数据的局部特征
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)

        # 状态空间模型（SSM）更新
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=x.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")

        # 规范化和输出投影
        y = self.norm(y, z)
        y = self.out_proj(y)

        # 更新隐藏状态
        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache):
        """
        推理单步函数，基于当前输入和隐藏状态执行单步推理。

        参数:
            u: Tensor - 当前输入，形状为(batch, 1, d_model)。
            h: InferenceCache - 初始/运行中的隐藏状态。

        返回:
            y: Tensor - 推理输出，形状为(batch, 1, d_model)。
            h: InferenceCache - 更新后的隐藏状态。
        """
        # 确保每次仅处理一个令牌
        assert u.shape[1] == 1, "每次推理步骤只能解码一个令牌"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # 更新卷积输入
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC

        # 卷积计算
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """
    执行稳定的段和计算。

    段和操作在这里的作用是将一个矩阵通过特定的累积求和方式，转换成1-半分离矩阵，这对于后续处理如时间序列建模等任务是有帮助的。
    该函数的命名灵感来源于'segment sum'的概念，但实现上有所不同，因此选择了一个新的命名以避免混淆。

    参数:
    - x: 输入的张量，通常代表一些需要转换的数据。
    - device: 可选参数，指定运算设备，如GPU或CPU。默认为None，表示使用默认设备。

    返回:
    - 一个经过特定方式累积求和转换后的张量。

    `exp(segsum(A))` 会产生一个1-半分离矩阵，这等价于一个标量SSM (Scalar Semi-Separable Matrix)。

    参考: 
    https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    
    # 获取输入张量x的最后一个维度的大小，通常代表时间序列的长度
    T = x.size(-1)
   
    # 将输入张量x沿着最后一个维度重复扩展，以便后续操作
    x = repeat(x, "... d -> ... d e", e=T)
    
    # 创建一个下三角掩码矩阵，用于后续操作中屏蔽不需要的元素
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    
    # 将输入张量x中不属于下三角的部分设置为0，实现屏蔽
    x = x.masked_fill(~mask, 0)
    
    # 沿着倒数第二个维度对x进行累积求和，这是计算段和的一部分
    x_segsum = torch.cumsum(x, dim=-2)
    
    # 创建一个新的下三角掩码矩阵，包括对角线，用于后续操作
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    
    # 将累积求和结果中不属于新的下三角的部分设置为负无穷，进行屏蔽
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    
    # 返回最终的段和结果
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """
    结构化状态空间对偶 (SSD) - Mamba-2 的核心算法

    这几乎是博客文章中的最小 SSD 代码。

    参数
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)
        
        chunk_size: 分块大小
        initial_states: 初始状态，默认为 None
        device: 设备，默认为 None

    返回
        y: (batch, seqlen, n_heads, d_head)
        final_state: 最终状态

    来源
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # 将输入数据重排成分块
    # SSD 的步骤 1、2 和 4 可以在不同设备上并行计算每个分块（序列并行）
    # 此功能未实现，留给读者自行完成
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. 计算每个分块内的输出（对角块）
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. 计算每个分块内的状态
    # （低秩分解的离对角块右项；B 项）
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. 计算分块间的 SSM 循环；产生分块边界上的正确 SSM 状态
    # （低秩分解的离对角块中间项；A 项）
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. 按分块计算状态到输出的转换
    # （低秩分解的离对角块左项；C 项）
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # 将分块内输出与分块间输出相加（对角块和离对角块）
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """
        初始化Gated Root Mean Square Layer Normalization模块。

        本模块实现了在论文《Gated Transformer》中提出的Gated Root Mean Square Layer Normalization方法。
        相较于传统的层归一化，该方法在某些场景下能提供更好的性能。

        参数:
        - d: int，特征维度，即归一化将要应用于的特征向量的长度。
        - eps: float，数值稳定性项，用于避免分母为零。默认值为1e-5。
        - device: Device，权重参数所使用的设备，用于支持在不同硬件设备上的运算。默认为None，意味着将使用CPU。

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps  # 定义用于数值稳定的epsilon值
        self.weight = nn.Parameter(torch.ones(d, device=device))  # 初始化权重参数，形状为(d,)

    def forward(self, x, z=None):
        """
        对输入张量x应用Gated Root Mean Square Layer Normalization。

        如果提供了额外的张量z，则首先对x进行门控操作，使用z的sigmoid作为门控函数。
        之后，对x进行归一化，然后乘以预定义的权重和一个epsilon值以确保数值稳定性。

        参数:
        - x: 输入张量，需要进行归一化的数据。
        - z: 可选张量，如果提供，则用于门控操作。

        返回:
        - 归一化并可能进行门控操作后的张量。
        """
        if z is not None:
            x = x * silu(z)  # 如果提供了z，则执行门控操作，使用silu函数作为激活函数
        # 对x进行归一化：首先计算x平方的平均值，然后进行倒数平方根操作，最后乘以权重和epsilon
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def silu(x):
    """
    应用Sigmoid Linear Unit（SiLU）激活函数，逐元素应用。

    由于PyTorch的实现在MPS（苹果的Metal Performance Shaders）上似乎无法工作，因此需要手动定义此函数来确保兼容性。
    
    参数:
    x (Tensor): 输入张量
    
    返回:
    Tensor: 应用SiLU激活函数后的输出张量
    """
    return x * F.sigmoid(x)

class BaseNdMamba2(nn.Module):
    """
    BaseNdMamba2类继承自nn.Module，用于实现一个特定的神经网络结构。
    该类的主要功能是通过Mamba2单元，对输入数据进行前向和后向处理，最终输出变换后的数据。
    
    参数:
    - cin: 输入数据的通道数。
    - cout: 输出数据的通道数。
    - mamba_dim: Mamba2单元的内部维度，必须是64的倍数。
    - **mamba2_args: Mamba2单元的额外参数，用于其初始化。
    
    返回:
    该构造器不返回值，但初始化了BaseNdMamba2类的实例。
    """
    def __init__(self, cin, cout, mamba_dim, **mamba2_args):
        super().__init__()
        # 确保mamba_dim是64的倍数，这对于后续计算的高效性和准确性至关重要。
        assert mamba_dim % 64 == 0, "cmid 必须是64的倍数"
        
        # 调整输入数据的通道数到mamba_dim，以便于后续的Mamba2单元处理。
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)
        
        # 初始化正向Mamba2单元，用于对数据进行前向处理。
        self.mamba2_for = Mamba2(mamba_dim, **mamba2_args)
        
        # 初始化负向Mamba2单元，用于对数据进行后向处理。
        self.mamba2_back = Mamba2(mamba_dim, **mamba2_args)
        
        # 调整经过Mamba2单元处理后的数据通道数到cout，作为最终输出。
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)


class NdMamba2_1d(BaseNdMamba2):
    """
    NdMamba2_1d类是BaseNdMamba2类的子类，用于实现特定的前向传播功能。
    它主要针对1D数据进行处理，通过继承BaseNdMamba2类的功能，并进行特定的调整和扩展。
    
    构造函数初始化了类的必要参数。
    
    参数:
    - cin: 输入通道数
    - cmid: 中间处理的通道数
    - cout: 输出通道数
    - **mamba2_args: 其他关键字参数，传递给BaseNdMamba2类
    """
    def __init__(self, cin, cmid, cout, **mamba2_args):
        super().__init__(cin, cmid, cout, **mamba2_args)

    def forward(self, x):
        """
        forward方法对输入数据x进行前向传播处理。
        它首先对输入数据进行填充，以确保数据长度为特定值的倍数，然后调整数据的维度，
        并通过NdMamba2网络进行正向和反向处理，最后再调整回原始维度并输出。
        
        参数:
        - x: 输入的数据，形状为[b, c, l]，其中b是批量大小，c是通道数，l是数据长度
        
        返回:
        - 经过处理的数据，形状调整为[b, c, l]，其中l是原始输入数据的长度
        """
        # 计算输入数据长度，用于后续的裁剪操作
        l = x.shape[2]
        # 将数据长度pad到64的倍数，以便于后续处理
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将 l , pad到4的倍数, [b, c64,l4]
        # 调整数据维度，从[b, c, l]到[b, l, c]，以适应后续的全连接层输入要求
        x = rearrange(x, 'b c l-> b l c')  # 转成 1d 信号 [b, d4*w4*h4, c64]
        # 通过全连接层调整通道数到合适的目标通道数
        x = self.fc_in(x)  # 调整通道数为目标通道数
        # 通过NdMamba2网络的正向传播处理
        x1, h1 = self.mamba2_for(x)
        # 通过NdMamba2网络的反向传播处理，并将结果翻转以恢复原始顺序
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        # 将正向和反向传播的结果相加，以融合信息
        x = x1 + x2
        # 通过全连接层调整通道数回到原始通道数
        x = self.fc_out(x)  # 调整通道数为目标通道数
        # 调整数据维度，从[b, l, c]到[b, c, l]，恢复到原始的数据格式
        x = rearrange(x, 'b l c -> b c l')  # 转成 2d 图片[b, l64, c64]
        # 裁剪数据，恢复到原始输入数据的长度
        x = x[:, :, :l]  # 截取原图大小
        return x


class NdMamba2_2d(BaseNdMamba2):
    """
    NdMamba2_2d类是BaseNdMamba2类的子类，用于实现特定的前向传播功能。
    它主要针对2D数据进行处理，通过继承BaseNdMamba2类的功能，并进行特定的调整和扩展。
    
    构造函数初始化了类的必要参数。
    
    参数:
    - cin: 输入通道数
    - cmid: 中间处理的通道数
    - cout: 输出通道数
    - **mamba2_args: 其他关键字参数，传递给BaseNdMamba2类
    """
    
    def __init__(self, cin,  cout,mamba_dim, **mamba2_args):
        # 继承BaseNdMamba2类，并初始化
        super().__init__(cin, cout, mamba_dim, **mamba2_args)

    def forward(self, x):
        # 获取输入特征图的高度和宽度
        h, w = x.shape[2:]
        # 将高度和宽度padding到8的倍数，以满足后续处理的需求
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8)
                  )
        # 获取padding后的高度和宽度
        h8, w8 = x.shape[2:]
        # 将特征图重新排列成1D信号，以便进行后续的全连接操作
        x = rearrange(x, 'b c h w -> b (h w) c')
        # 调整通道数为目标通道数，为后续的Mamba2操作做准备
        x = self.fc_in(x)
        # 使用Mamba2前向处理，并获得输出和隐藏状态
        x1, h1 = self.mamba2_for(x)
        # 对输入进行翻转后，使用Mamba2后向处理，并获得输出和隐藏状态
        x2, h2 = self.mamba2_back(x.flip(1))
        # 将后向处理的输出翻转回来，以恢复原始顺序
        x2 = x2.flip(1)
        # 将前向和后向的处理结果相加
        x = x1 + x2
        # 调整通道数为目标通道数，完成最后的输出调整
        x = self.fc_out(x)
        # 将特征图重新排列成2D图片，恢复到padding后的尺寸
        x = rearrange(x, 'b (h w) c -> b c h w', h=h8)
        # 截取特征图，恢复到原始输入的尺寸
        x = x[:, :, :h, :w]
        # 返回处理后的特征图
        return x


class NdMamba2_3d(BaseNdMamba2):
    """
    NdMamba2_3d类是BaseNdMamba2类的子类，用于实现特定的前向传播功能。
    它主要针对3D数据进行处理，通过继承BaseNdMamba2类的功能，并进行特定的调整和扩展。
    
    构造函数初始化了类的必要参数。
    
    参数:
    - cin: 输入通道数
    - cmid: 中间处理的通道数
    - cout: 输出通道数
    - **mamba2_args: 其他关键字参数，传递给BaseNdMamba2类
    """
    
    def __init__(self, cin,  cout,mamba_dim, **mamba2_args):
        # 继承基类并进行初始化
        super().__init__(cin, cout, mamba_dim, **mamba2_args)

    def forward(self, x):
        # 获取输入张量的空间维度值
        d, h, w = x.shape[2:]
        # 对输入张量进行填充，使其空间维度成为4的倍数
        x = F.pad(x, (0, (4 - x.shape[4] % 4) % 4,
                      0, (4 - x.shape[3] % 4) % 4,
                      0, (4 - x.shape[2] % 4) % 4)
                  )  # 将 d, h, w , pad到4的倍数, [b, c64,d4, h4, w4]
        # 再次获取填充后张量的空间维度值
        d4, h4, w4 = x.shape[2:]
        # 改变张量形状，将其从3D转换为1D信号
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # 转成 1d 信号 [b, d4*w4*h4, c64]
        # 通过全连接层调整输入张量的通道数
        x = self.fc_in(x)  # 调整通道数为目标通道数
        # 正向传播通过Mamba2模块
        x1, h1 = self.mamba2_for(x)
        # 反向传播通过Mamba2模块，同时翻转序列
        x2, h2 = self.mamba2_back(x.flip(1))
        # 翻转反向传播的输出，恢复原始顺序
        x2 = x2.flip(1)
        # 将正向和反向传播的结果相加
        x = x1 + x2
        # 通过全连接层调整输出张量的通道数
        x = self.fc_out(x)  # 调整通道数为目标通道数
        # 将张量形状从1D信号转换回2D图片
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d4, h=h4, w=w4)  # 转成 2d 图片[b, d4*w4*h4, c64]
        # 截取张量，恢复到原始输入的大小
        x = x[:, :, :d, :h, :w]  # 截取原图大小
        # 返回处理后的张量
        return x


class NdMamba2(BaseNdMamba2):
    """
    NdMamba2类是基于BaseNdMamba2的，用于实现更复杂的数据处理。
    
    其构造函数接受以下参数：
    - cin: 输入通道数
    - cout: 输出通道数
    - mamba_dim: 特征维度
    - **mamba2_args: 其他关键参数，用于初始化BaseNdMamba2类
    """
    def __init__(self, cin,  cout, mamba_dim, **mamba2_args):
        super().__init__(cin, cout, mamba_dim, **mamba2_args)

    def forward(self, x):
        """
        forward方法对输入数据x进行处理。
        - x: 输入的多维张量，其形状为(batch_size, cin, *size)
        
        返回处理后的张量，形状与输入张量相同。
        
        该方法的具体处理步骤如下：
        1. 获取输入张量x的尺寸信息，用于后续恢复形状。
        2. 将x展平为二维张量，以便进行处理。
        3. 对展平后的张量x进行填充，使其形状的最后一个维度为64的倍数。
        4. 重新排列张量x的维度顺序，以便进行一维信号处理。
        5. 使用fc_in层调整张量x的通道数。
        6. 分别使用mamba2_for和mamba2_back对张量x进行正向和反向处理。
        7. 将正向和反向处理的结果相加，并使用fc_out层再次调整通道数。
        8. 恢复张量的原始形状，返回处理后的张量。
        """
        
        # 获取输入张量的尺寸信息
        size = x.shape[2:]
        # 将输入张量展平为二维张量
        x = torch.flatten(x, 2)
        # 计算展平后张量的长度
        l = x.shape[2]
        # 对展平后的张量进行填充，使其长度为64的倍数
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))
        # 重新排列张量的维度顺序，以便进行一维信号处理
        x = rearrange(x, 'b c l-> b l c')
        # 使用fc_in层调整张量的通道数
        x = self.fc_in(x)
        # 使用mamba2_for对张量进行正向处理
        x1, h1 = self.mamba2_for(x)
        # 使用mamba2_back对张量进行反向处理，并将结果翻转
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        # 将正向和反向处理的结果相加
        x = x1 + x2
        # 使用fc_out层再次调整张量的通道数
        x = self.fc_out(x)
        # 恢复张量的原始形状
        x = rearrange(x, 'b l c -> b c l')
        # 截取张量至原图大小
        x = x[:, :, :l]
        # 根据原始尺寸信息恢复张量的形状
        x = torch.unflatten(x, 2, size)
        return x

if __name__ == '__main__':
    # 通用的多维度双向mamba2
    net_n = NdMamba2(64, 128, 64).cuda()

    # 定制的双向mamba2 1d, 2d, 3d
    net1 = NdMamba2_1d(64, 128, 64).cuda()
    net2 = NdMamba2_2d(64, 128, 64).cuda()
    net3 = NdMamba2_3d(64, 128, 64).cuda()

    # 多维度数据
    x1 = torch.randn(1, 64, 32).cuda() # 1d
    x2 = torch.randn(1, 64, 32, 77).cuda() # 2d
    x3 = torch.randn(1, 64, 32, 77, 25).cuda() # 3d
    x4 = torch.randn(1, 64, 32, 77, 25, 15).cuda() # 4d

    # 测试
    y1 = net_n(x1)
    print(y1.shape)
    y2 = net_n(x2)
    print(y2.shape)
    y3 = net_n(x3)
    print(y3.shape)
    y4 = net_n(x4)
    print(y4.shape)


    y1 = net1(x1)
    print(y1.shape)
    y2 = net2(x2)
    print(y2.shape)
    y3 = net3(x3)
    print(y3.shape)

