
"""
================================================================================
Transformer 完整复现与消融实验
论文: "Attention Is All You Need" (Vaswani et al., 2017)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 第一部分: 基础组件
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (标准版本)
    公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影矩阵 W^Q, W^K, W^V, W^O
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性投影并分头: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 3. 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. 加权求和
        context = torch.matmul(attn_weights, V)

        # 6. 拼接多头并线性投影
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ==============================================================================
# 第二部分: 位置编码变体 (实验2.1)
# ==============================================================================

class SinusoidalPosEnc(nn.Module):
    """原文方法: 正弦/余弦位置编码"""
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AbsolutePosEnc(nn.Module):
    """简单绝对位置编码: 可学习的Embedding"""
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(x.size(0), x.size(1))
        x = x + self.pe(positions)
        return self.dropout(x)


class NoPosEnc(nn.Module):
    """无位置编码 (基线对比)"""
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x)


# ==============================================================================
# 第三部分: Q,K,V变体 (实验2.2)
# ==============================================================================

class QKOnlyAttention(nn.Module):
    """Q,K,V变体: K和V合并为同一个矩阵"""
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)  # K和V共享
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = K.clone()  # V直接使用K的值

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out), attn


# ==============================================================================
# 第四部分: 残差连接变体 (实验2.3)
# ==============================================================================

class EncoderLayer(nn.Module):
    """标准编码器层 (含残差连接)"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))  # 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))      # 残差连接
        return x


class EncoderLayerNoRes(nn.Module):
    """编码器层: 去掉残差连接"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(self.dropout1(attn_output))  # 无残差
        x = self.norm2(self.dropout2(self.feed_forward(x)))  # 无残差
        return x


class DecoderLayer(nn.Module):
    """标准解码器层 (含残差连接)"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x


class DecoderLayerNoRes(nn.Module):
    """解码器层: 去掉残差连接"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(self.dropout1(attn_output))
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(self.dropout2(attn_output))
        x = self.norm3(self.dropout3(self.feed_forward(x)))
        return x


# ==============================================================================
# 第五部分: CNN替代方案 (实验2.4)
# ==============================================================================

class CNNSeq2Seq(nn.Module):
    """用CNN重构序列到序列模型 + 位置编码"""
    def __init__(self, src_v=100, tgt_v=100, d=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.d = d
        self.src_emb = nn.Embedding(src_v, d)
        self.tgt_emb = nn.Embedding(tgt_v, d)
        self.scale = math.sqrt(d)

        # 位置编码
        self.pos = SinusoidalPosEnc(d, 5000, dropout)

        # 编码器: 1D CNN + Gating
        self.enc_convs = nn.ModuleList()
        for _ in range(n_layers):
            self.enc_convs.append(nn.Sequential(
                nn.Conv1d(d, d * 2, kernel_size=3, padding=1),
                nn.GLU(dim=1),
                nn.Dropout(dropout)
            ))

        # 解码器: 1D CNN + Gating
        self.dec_convs = nn.ModuleList()
        for _ in range(n_layers):
            self.dec_convs.append(nn.Sequential(
                nn.Conv1d(d, d * 2, kernel_size=3, padding=1),
                nn.GLU(dim=1),
                nn.Dropout(dropout)
            ))

        self.out = nn.Linear(d, tgt_v)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        # 编码器 + 位置编码
        x = self.src_emb(src) * self.scale
        x = self.pos(x)
        x = x.permute(1, 2, 0)  # (seq, batch, d) -> (batch, d, seq)
        for conv in self.enc_convs:
            x = conv(x)
        enc = x.permute(2, 0, 1)  # (seq, batch, d)

        # 解码器 + 位置编码
        x = self.tgt_emb(tgt) * self.scale
        x = self.pos(x)
        x = x.permute(1, 2, 0)
        for conv in self.dec_convs:
            x = conv(x)
        x = x.permute(2, 0, 1)

        return self.out(x)


# ==============================================================================
# 第六部分: 通用Transformer框架 (支持所有变体)
# ==============================================================================

class GenericTransformer(nn.Module):
    """
    通用Transformer，可切换:
    - 位置编码类型 (sinusoidal/absolute/none)
    - 注意力机制 (standard/qk_only)
    - 是否使用残差连接
    """
    def __init__(self, src_v=100, tgt_v=100, d=256, h=8, n_enc=3, n_dec=3, 
                 d_ff=1024, drop=0.1, pos_enc_type='sinusoidal', 
                 attn_type='standard', use_residual=True):
        super().__init__()
        self.d = d
        self.src_emb = nn.Embedding(src_v, d)
        self.tgt_emb = nn.Embedding(tgt_v, d)
        self.scale = math.sqrt(d)
        self.use_residual = use_residual

        # 位置编码选择
        if pos_enc_type == 'sinusoidal':
            self.pos = SinusoidalPosEnc(d, 5000, drop)
        elif pos_enc_type == 'absolute':
            self.pos = AbsolutePosEnc(d, 5000, drop)
        else:
            self.pos = NoPosEnc(d, 5000, drop)

        # 注意力机制选择
        AttnClass = MultiHeadAttention if attn_type == 'standard' else QKOnlyAttention

        # 编码器/解码器层选择
        if use_residual:
            from types import SimpleNamespace
            # 使用标准层
            self.encs = nn.ModuleList([
                EncoderLayer(d, h, d_ff, drop) for _ in range(n_enc)
            ])
            self.decs = nn.ModuleList([
                DecoderLayer(d, h, d_ff, drop) for _ in range(n_dec)
            ])
        else:
            self.encs = nn.ModuleList([
                EncoderLayerNoRes(d, h, d_ff, drop) for _ in range(n_enc)
            ])
            self.decs = nn.ModuleList([
                DecoderLayerNoRes(d, h, d_ff, drop) for _ in range(n_dec)
            ])

        self.out = nn.Linear(d, tgt_v)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.src_emb(src) * self.scale
        x = self.pos(x)
        for layer in self.encs:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.tgt_emb(tgt) * self.scale
        x = self.pos(x)
        for layer in self.decs:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.out(decoder_output)


# ==============================================================================
# 第七部分: 辅助函数
# ==============================================================================

def generate_padding_mask(seq, pad_idx=0):
    """生成padding mask"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_look_ahead_mask(size):
    """生成上三角mask，防止解码器看到未来token"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def generate_masks(src, tgt, pad_idx=0):
    """生成所有需要的mask"""
    src_mask = generate_padding_mask(src, pad_idx)
    tgt_pad_mask = generate_padding_mask(tgt, pad_idx)
    tgt_look_ahead = generate_look_ahead_mask(tgt.size(1)).to(tgt.device)
    tgt_mask = tgt_pad_mask & tgt_look_ahead.unsqueeze(0).unsqueeze(0)
    return src_mask, tgt_mask


def create_copy_task_data(batch_size=32, seq_len=20, vocab_size=100):
    """生成复制任务数据"""
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = src.clone()
    bos = torch.ones(batch_size, 1, dtype=torch.long)
    tgt_input = torch.cat([bos, tgt[:, :-1]], dim=1)
    tgt_output = tgt
    return src, tgt_input, tgt_output


def greedy_decode(model, src, max_len=11, bos_idx=1, pad_idx=0):
    """贪心解码 (用于Transformer)"""
    model.eval()
    batch_size = src.size(0)
    src_mask = generate_padding_mask(src, pad_idx)
    encoder_output = model.encode(src, src_mask)
    tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_len - 1):
            _, tgt_mask = generate_masks(src, tgt, pad_idx)
            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            next_token = model.out(output)[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

    return tgt


def greedy_decode_cnn(model, src, max_len=11, bos_idx=1):
    """贪心解码 (用于CNN)"""
    model.eval()
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(src, tgt)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

    return tgt


def train_model(model, data_src, data_tin, data_tout, epochs=50, bs=512, lr=1e-2, is_cnn=False):
    """通用训练函数"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        idx = torch.randint(0, len(data_src), (bs,))
        src = data_src[idx]
        tin = data_tin[idx]
        tout = data_tout[idx]

        optimizer.zero_grad()
        if is_cnn:
            logits = model(src, tin)
        else:
            src_mask, tgt_mask = generate_masks(src, tin, 0)
            logits = model(src, tin, src_mask, tgt_mask)

        loss = criterion(logits.view(-1, logits.size(-1)), tout.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return losses


def evaluate_model(model, test_src, test_tout, is_cnn=False):
    """评估准确率"""
    if is_cnn:
        pred = greedy_decode_cnn(model, test_src, max_len=11)
    else:
        pred = greedy_decode(model, test_src, max_len=11)
    correct = (pred[:, 1:] == test_tout).float().sum().item()
    total = test_tout.numel()
    return correct / total


# ==============================================================================
# 第八部分: 实验主程序
# ==============================================================================

def run_all_experiments():
    """运行所有消融实验"""

    # 生成固定数据
    vocab_size = 20
    fixed_src = torch.randint(1, 11, (1000, 10))
    fixed_tout = fixed_src.clone()
    fixed_tin = torch.cat([torch.ones(1000, 1, dtype=torch.long), fixed_tout[:, :-1]], dim=1)
    test_src = torch.randint(1, 11, (200, 10))
    test_tout = test_src.clone()

    results = {}

    # 实验2.1: 位置编码对比
    print("="*60)
    print("实验 2.1: 位置编码对比")
    print("="*60)
    for pos_type, name in [('sinusoidal', '正弦编码'), ('absolute', '绝对编码'), ('none', '无编码')]:
        model = GenericTransformer(src_v=vocab_size, tgt_v=vocab_size, d=32, h=2, n_enc=1, n_dec=1,
                                    d_ff=128, drop=0.1, pos_enc_type=pos_type)
        losses = train_model(model, fixed_src, fixed_tin, fixed_tout, epochs=50)
        train_acc = evaluate_model(model, fixed_src[:200], fixed_tout[:200])
        test_acc = evaluate_model(model, test_src, test_tout)
        results[f'pos_{pos_type}'] = {'losses': losses, 'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name}: Loss={losses[-1]:.3f} Train={train_acc*100:.1f}% Test={test_acc*100:.1f}%")

    # 实验2.2: Q,K,V必要性
    print("\n" + "="*60)
    print("实验 2.2: Q,K,V必要性")
    print("="*60)
    for attn_type, name in [('standard', '标准Q,K,V'), ('qk_only', 'Q,K共享')]:
        model = GenericTransformer(src_v=vocab_size, tgt_v=vocab_size, d=32, h=2, n_enc=1, n_dec=1,
                                    d_ff=128, drop=0.1, attn_type=attn_type)
        losses = train_model(model, fixed_src, fixed_tin, fixed_tout, epochs=50)
        train_acc = evaluate_model(model, fixed_src[:200], fixed_tout[:200])
        test_acc = evaluate_model(model, test_src, test_tout)
        results[f'attn_{attn_type}'] = {'losses': losses, 'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name}: Loss={losses[-1]:.3f} Train={train_acc*100:.1f}% Test={test_acc*100:.1f}%")

    # 实验2.3: 残差连接
    print("\n" + "="*60)
    print("实验 2.3: 残差连接必要性")
    print("="*60)
    for use_res, name in [(True, '有残差'), (False, '无残差')]:
        model = GenericTransformer(src_v=vocab_size, tgt_v=vocab_size, d=32, h=2, n_enc=1, n_dec=1,
                                    d_ff=128, drop=0.1, use_residual=use_res)
        losses = train_model(model, fixed_src, fixed_tin, fixed_tout, epochs=50)
        train_acc = evaluate_model(model, fixed_src[:200], fixed_tout[:200])
        test_acc = evaluate_model(model, test_src, test_tout)
        results[f'res_{use_res}'] = {'losses': losses, 'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name}: Loss={losses[-1]:.3f} Train={train_acc*100:.1f}% Test={test_acc*100:.1f}%")

    # 实验2.4: CNN vs Transformer
    print("\n" + "="*60)
    print("实验 2.4: CNN vs Transformer")
    print("="*60)
    cnn_model = CNNSeq2Seq(src_v=vocab_size, tgt_v=vocab_size, d=32, n_layers=2, dropout=0.1)
    cnn_losses = train_model(cnn_model, fixed_src, fixed_tin, fixed_tout, epochs=50, is_cnn=True)
    cnn_train_acc = evaluate_model(cnn_model, fixed_src[:200], fixed_tout[:200], is_cnn=True)
    cnn_test_acc = evaluate_model(cnn_model, test_src, test_tout, is_cnn=True)
    results['cnn'] = {'losses': cnn_losses, 'train_acc': cnn_train_acc, 'test_acc': cnn_test_acc}
    print(f"CNN+PosEnc: Loss={cnn_losses[-1]:.3f} Train={cnn_train_acc*100:.1f}% Test={cnn_test_acc*100:.1f}%")

    trans_model = GenericTransformer(src_v=vocab_size, tgt_v=vocab_size, d=32, h=2, n_enc=2, n_dec=2,
                                     d_ff=128, drop=0.1)
    trans_losses = train_model(trans_model, fixed_src, fixed_tin, fixed_tout, epochs=50)
    trans_train_acc = evaluate_model(trans_model, fixed_src[:200], fixed_tout[:200])
    trans_test_acc = evaluate_model(trans_model, test_src, test_tout)
    results['transformer'] = {'losses': trans_losses, 'train_acc': trans_train_acc, 'test_acc': trans_test_acc}
    print(f"Transformer: Loss={trans_losses[-1]:.3f} Train={trans_train_acc*100:.1f}% Test={trans_test_acc*100:.1f}%")

    return results


if __name__ == '__main__':
    results = run_all_experiments()
