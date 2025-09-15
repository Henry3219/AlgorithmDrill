import torch
# torch.nn 是 PyTorch 中包含所有神经网络层、激活函数和损失函数等构建块的模块
import torch.nn as nn
import math

# 定义多头注意力（MHA）模块。
# nn.Module 是 PyTorch 中所有神经网络模块的基类。我们自定义的任何模型或层都应该继承这个类。
class MHA(nn.Module):
    # 类的构造函数（初始化方法）
    # num_head: 注意力头的数量
    # dimension_k: 输入的键（Key）和查询（Query）的特征维度
    # dimension_v: 输入的值（Value）的特征维度
    # d_k: 每个注意力头中，键和查询被投影到的维度
    # d_v: 每个注意力头中，值被投影到的维度
    # d_o: 最终输出的特征维度
    # dropout: Dropout的比率，用于防止过拟合
    def __init__(self, num_head, dimension_k, dimension_v, d_k, d_v, d_o, dropout=0.1):
        # 必须调用父类 nn.Module 的构造函数
        super().__init__()
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_o

        # nn.Linear 是一个全连接层，它对输入进行线性变换 y = xA^T + b
        # 这里我们用它来将输入的 Q, K, V 投影到多头所需的总维度上
        # 之所以叫 fc 是 Fully Connected（全连接）的缩写
        self.fc_q = nn.Linear(dimension_k, num_head * d_k)
        self.fc_k = nn.Linear(dimension_k, num_head * d_k)
        self.fc_v = nn.Linear(dimension_v, num_head * d_v)

        # nn.Dropout 是一个正则化层，在训练期间以一定概率（dropout）将输入张量中的元素随机置为零
        self.dropout = nn.Dropout(dropout)
        # 最后的输出层，也是一个全连接层，将拼接后的多头结果映射到最终的输出维度
        self.fc_o = nn.Linear(num_head * d_v, d_o)
        
        # nn.Softmax 是一个激活函数，它将一个实数向量压缩成一个概率分布（所有元素和为1）
        # dim=-1 表示在最后一个维度上进行 Softmax 操作
        self.softmax = nn.Softmax(dim=-1)

    # 前向传播函数，定义了当数据输入模块时所执行的计算
    def forward(self, q, k, v, mask):
        # 初始输入形状:
        # q: [b, n_q, dimension_k] (b=batch_size, n_q=query序列长度)
        # k: [b, n_k, dimension_k] (n_k=key序列长度)
        # v: [b, n_v, dimension_v] (n_v=value序列长度, 通常 n_k == n_v)
        # mask: [b, n_q, n_k]
        
        # 获取批次大小和序列长度
        b, n_q, _ = q.size()
        _, n_k, _ = k.size()
        _, n_v, _ = v.size()
        h = self.num_head

        # 1. 线性映射: 将输入 q, k, v 投影到高维空间，这个空间包含了所有头的信息
        # q: [b, n_q, dimension_k] -> [b, n_q, h * d_k]
        q = self.fc_q(q)
        # k: [b, n_k, dimension_k] -> [b, n_k, h * d_k]
        k = self.fc_k(k)
        # v: [b, n_v, dimension_v] -> [b, n_v, h * d_v]
        v = self.fc_v(v)

        # 2. 拆分多头: 将投影后的 q, k, v 拆分成 h 个独立的头
        # .view() 用于重塑张量形状，-1 表示该维度大小由其他维度自动推断
        # .transpose(1, 2) 用于交换第1和第2个维度，目的是将 'num_head' 维度提前，方便并行计算
        # Q: [b, n_q, h * d_k] -> view -> [b, n_q, h, d_k] -> transpose -> [b, h, n_q, d_k]
        Q = q.view(b, n_q, h, self.d_k).transpose(1,2)
        # K: [b, n_k, h * d_k] -> view -> [b, n_k, h, d_k] -> transpose -> [b, h, n_k, d_k]
        K = k.view(b, n_k, h, self.d_k).transpose(1,2)
        # V: [b, n_v, h * d_v] -> view -> [b, n_v, h, d_v] -> transpose -> [b, h, n_v, d_v]
        V = v.view(b, n_v, h, self.d_v).transpose(1,2)

        # 3. 计算注意力分数 (Scaled Dot-Product Attention)
        # K.transpose(-1, -2) 将 K 的最后两个维度交换，形状变为 [b, h, d_k, n_k]
        # torch.matmul(Q, K^T) 计算 Q 和 K^T 的点积，得到注意力原始分数
        # 形状: [b, h, n_q, d_k] @ [b, h, d_k, n_k] -> [b, h, n_q, n_k]
        # 除以 math.sqrt(self.d_k) 是进行缩放，防止梯度过大或过小
        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_k)

        # 4. 应用掩码 (Masking)
        # mask 初始形状: [b, n_q, n_k]
        # .unsqueeze(1) 在第1维增加一个维度，使其能与 scores 的多头维度进行广播
        # 形状: [b, n_q, n_k] -> [b, 1, n_q, n_k]
        mask = mask.unsqueeze(1)
        # 将 mask 加到 scores 上。mask 中为 -inf 的位置在 softmax 后会变为 0，实现屏蔽效果
        # scores 形状: [b, h, n_q, n_k] (mask被广播到h个头上)
        scores = scores + mask

        # 5. 计算注意力权重
        # 对 scores 在最后一个维度(n_k)上应用 softmax，得到归一化的注意力权重
        # 形状不变: [b, h, n_q, n_k]
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        # 6. 加权求和
        # 将注意力权重 attn 与 V 进行矩阵乘法，得到每个头的输出
        # 形状: [b, h, n_q, n_k] @ [b, h, n_v, d_v] -> [b, h, n_q, d_v] (假设 n_k == n_v)
        head_out = torch.matmul(attn, V)

        # 7. 合并多头
        # .transpose(1, 2) 换回维度，为拼接做准备。形状: [b, n_q, h, d_v]
        # .contiguous() 确保张量在内存中是连续的，这是 .view() 操作的要求
        # .view() 将所有头拼接在一起。形状: [b, n_q, h * d_v]
        head_out = head_out.transpose(1,2).contiguous().view(b, n_q, h * self.d_v)

        # 8. 最终线性投影
        # 通过输出线性层得到最终的输出
        # 形状: [b, n_q, h * d_v] -> [b, n_q, d_o]
        out = self.fc_o(head_out)
        
        return attn, out

# 主代码
batch = 10
num_head = 8
n_q, n_k, n_v = 4, 4, 4
dimension_q = dimension_k = 128
dimension_v = 64
d_k, d_v, d_o = 16, 16, 8

q = torch.randn(batch, n_q, dimension_q)
k = torch.randn(batch, n_k, dimension_k)
v = torch.randn(batch, n_v, dimension_v)

# 构造一个上三角mask
mask = torch.full((batch, n_q, n_k),float('-inf'))
mask = torch.triu(mask, diagonal = 1)

mha = MHA(num_head, dimension_k, dimension_v, d_k, d_v, d_o)
attention, out = mha(q, k, v, mask)

print(attention.size(), out.size())






