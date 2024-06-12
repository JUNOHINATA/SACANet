import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Multi-Branch SelfAttention
# class MultiHeadSelfAttentionBlock(nn.Module):
#     def __init__(self, input_size, hidden_size, num_heads):
#         super(MultiHeadSelfAttentionBlock, self).__init__()
#         assert hidden_size % num_heads == 0, "hidden size should be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_size = hidden_size // num_heads

#         self.query = nn.Linear(input_size, hidden_size)
#         self.key = nn.Linear(input_size, hidden_size)
#         self.value = nn.Linear(input_size, hidden_size)
#         self.output_layer = nn.Linear(hidden_size, input_size)

#     def forward(self, x):
#         batch_size, input_size = x.size()  # 修改这里，移除对seq_len的引用
#         q = self.query(x).view(batch_size, self.num_heads, self.head_size).transpose(1, 0)
#         k = self.key(x).view(batch_size, self.num_heads, self.head_size).transpose(1, 0)
#         v = self.value(x).view(batch_size, self.num_heads, self.head_size).transpose(1, 0)

#         attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5), dim=-1)
#         x = torch.matmul(attn_weights, v).transpose(1, 0).contiguous().view(batch_size, -1)
#         x = self.output_layer(x)
#         return x


# class FullyConnected(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None, num_heads=8):
#         super(FullyConnected, self).__init__()
        
#         # 定义三个自注意力特征提取器
#         self.attention1 = MultiHeadSelfAttentionBlock(input_size, hidden_size, num_heads)
#         self.attention2 = MultiHeadSelfAttentionBlock(input_size, hidden_size, num_heads)
#         self.attention3 = MultiHeadSelfAttentionBlock(input_size, hidden_size, num_heads)
        
#         # 定义融合后的全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(258, hidden_size),  # 修改这里的输入特征数
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_size, output_size)
#         )

#     def forward(self, x):
#         # 通过三个自注意力特征提取器
#         x1 = self.attention1(x)
#         x2 = self.attention2(x)
#         x3 = self.attention3(x)
        
#         # 将输出特征进行融合
#         x = torch.cat((x1, x2, x3), dim=1)
        
#         # 通过全连接层处理融合后的特征
#         x = self.fc(x)
        
#         return x



# class GeoMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_geodesic, geodesic, dropout=0.0, batch_first=False):
#         super(GeoMultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_geodesic = num_geodesic
#         self.geodesic = geodesic  # geodesic matrix
#         self.dropout = dropout
#         self.batch_first = batch_first

#         # Initialize other necessary parameters and layers for MultiheadAttention
#         self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
#         self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
#         self.out_proj = nn.Linear(embed_dim, embed_dim)  # Output projection layer

#     def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
#         # Apply linear projections using matrix multiplication
#         q = torch.matmul(self.in_proj_weight[:self.embed_dim, :self.embed_dim], query)
#         k = torch.matmul(self.in_proj_weight[self.embed_dim:2 * self.embed_dim, :self.embed_dim], key)
#         v = torch.matmul(self.in_proj_weight[2 * self.embed_dim:, :self.embed_dim], value)

#         # Add the biases
#         q += self.in_proj_bias[:self.embed_dim].unsqueeze(0).unsqueeze(2)
#         k += self.in_proj_bias[self.embed_dim:2 * self.embed_dim].unsqueeze(0).unsqueeze(2)
#         v += self.in_proj_bias[2 * self.embed_dim:].unsqueeze(0).unsqueeze(2)

#         # Split and reshape for multi-head attention
#         q = q.view(query.size(0), query.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
#         k = k.view(key.size(0), key.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
#         v = v.view(value.size(0), value.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

#         # Apply attention and combine heads
#         attn_output, attn_output_weights = F.multi_head_attention_forward(
#             query=q,
#             key=k,
#             value=v,
#             num_heads=self.num_heads,
#             in_proj_weight=self.in_proj_weight,
#             in_proj_bias=self.in_proj_bias,
#             out_proj_weight=self.out_proj.weight,
#             out_proj_bias=self.out_proj.bias,
#             dropout_p=self.dropout,
#             training=self.training,
#             key_padding_mask=key_padding_mask,
#             need_weights=need_weights,
#             attn_mask=attn_mask,
#             use_separate_proj_weight=True,
#             q_proj_weight=self.in_proj_weight[:self.embed_dim],
#             k_proj_weight=self.in_proj_weight[self.embed_dim:2 * self.embed_dim],
#             v_proj_weight=self.in_proj_weight[2 * self.embed_dim:],
#         )

#         # If num_geodesic is not zero, apply the geodesic transformation
#         if self.num_geodesic > 0:
#             # Apply the geodesic matrix to the value tensor
#             n_dim_replace = self.embed_dim * self.num_geodesic // self.num_heads
#             p_value = value[:, :, :n_dim_replace]
#             p_value = p_value.permute(1, 0, 2)
#             new_value = torch.bmm(self.geodesic, p_value)
#             new_value = new_value.permute(1, 0, 2)
#             # Replace the corresponding part in the value tensor with the transformed values
#             value[:, :, :n_dim_replace] = new_value

#         # Apply dropout if necessary
#         if self.dropout > 0:
#             attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

#         # Return the modified attention output and optional weights
#         return attn_output, attn_output_weights

    # def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
    #     # Apply linear projections
    #     print("Shape of in_proj_weight:", self.in_proj_weight[:self.embed_dim].view(-1, self.embed_dim).shape)
    #     print("Shape of query:", query.transpose(0, 1).shape)

    #     q = self.in_proj_weight[:self.embed_dim].view(-1, self.embed_dim) @ query.transpose(0, 1) + self.in_proj_bias[:self.embed_dim]
    #     k = self.in_proj_weight[self.embed_dim:2 * self.embed_dim].view(-1, self.embed_dim) @ key.transpose(0, 1) + self.in_proj_bias[self.embed_dim:2 * self.embed_dim]
    #     v = self.in_proj_weight[2 * self.embed_dim:].view(-1, self.embed_dim) @ value.transpose(0, 1) + self.in_proj_bias[2 * self.embed_dim:]

    #     # Split and reshape for multi-head attention
    #     q = q.view(query.size(0), query.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
    #     k = k.view(key.size(0), key.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
    #     v = v.view(value.size(0), value.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

    #     # Apply attention and combine heads
    #     attn_output, attn_output_weights = F.multi_head_attention_forward(
    #         query=q,
    #         key=k,
    #         value=v,
    #         num_heads=self.num_heads,
    #         in_proj_weight=self.in_proj_weight,
    #         in_proj_bias=self.in_proj_bias,
    #         out_proj_weight=self.out_proj.weight,
    #         out_proj_bias=self.out_proj.bias,
    #         dropout_p=self.dropout,
    #         training=self.training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         use_separate_proj_weight=True,
    #         q_proj_weight=self.in_proj_weight[:self.embed_dim],
    #         k_proj_weight=self.in_proj_weight[self.embed_dim:2 * self.embed_dim],
    #         v_proj_weight=self.in_proj_weight[2 * self.embed_dim:],
    #     )
    #  # If num_geodesic is not zero, apply the geodesic transformation
    #     if self.num_geodesic > 0:
    #         # Apply the geodesic matrix to the value tensor
    #         # Assuming 'value' has shape (batch_size, sequence_length, embed_dim)
    #         # 'geodesic' has shape (num_geodesic, embed_dim)
    #         # The output will have shape (batch_size, sequence_length, num_geodesic * embed_dim // num_heads)
    #          # Apply the geodesic matrix to the value tensor
    #         # Assuming 'value' has shape (batch_size, sequence_length, embed_dim)
    #         # 'geodesic' has shape (num_geodesic, embed_dim)
    #         # The output will have shape (batch_size, sequence_length, num_geodesic * embed_dim // num_heads)
    #         # where num_geodesic * embed_dim // num_heads is the number of dimensions to replace
    #         n_dim_replace = self.embed_dim * self.num_geodesic // self.num_heads
    #         p_value = value[:, :, :n_dim_replace]
    #         p_value = p_value.permute(1, 0, 2)
    #         new_value = torch.bmm(self.geodesic, p_value)
    #         new_value = new_value.permute(1, 0, 2)
    #         # Replace the corresponding part in the value tensor with the transformed values
    #         value[:, :, :n_dim_replace] = new_value
    #         # Apply dropout if necessary
    #     if self.dropout > 0:
    #         attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

    #     # Return the modified attention输出 and option的权重
    #     # Return the modified attention输出 and option的权重
    #     return attn_output, attn_output_weights
    



# # ... (h_sigmoid 和 h_swish 类定义保持不变)
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         return self.relu(x + 3) / 6

# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)

#     def forward(self, x):
#         return x * self.sigmoid(x)


# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((1, None))
#         self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

#         mip = max(8, inp // reduction)

#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
        
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

#     def forward(self, x):
#         identity = x
        
#         n,c = x.size()
#         x_h = self.pool_h(x).view(n, c)
#         x_w = self.pool_w(x).view(n, c)

#         y = torch.cat([x_h, x_w], dim=1)
#         y = self.conv1(y.unsqueeze(-1).unsqueeze(-1))
#         y = self.bn1(y)
#         y = self.act(y) 
        
#         x_h, x_w = torch.split(y, [1, 1], dim=1)
#         x_w = x_w.permute(0, 2, 1)

#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()

#         out = identity * a_w * a_h

#         return out.view(n, c)

# class FullyConnected(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None, use_coord_att=True):
#         super(FullyConnected, self).__init__()
#         self.use_coord_att = use_coord_att
#         net = [
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.2),
#         ]
#         for i in range(num_layers - 2):
#             net.extend([
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(inplace=True),
#             ])
#         if use_coord_att:
#             net.append(CoordAtt(hidden_size, hidden_size))
#         net.extend([
#             nn.Linear(hidden_size, output_size),
#         ])
#         self.net = nn.Sequential(*net)

#     def forward(self, x):
#         return self.net(x)

# 示例 usage:
# fc = FullyConnected(input_size=128, output_size=10, hidden_size=512, num_layers=3, use_coord_att=True)
# input_data = torch.randn(32, 128)  # 假设的输入数据
# output_data = fc(input_data)


class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 保持输入特征维度为86
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # print("x shape before XCA forward: ", x.shape)
        B, N, C = x.shape

        qkv = self.qkv(x)
        # print("qkv shape:", qkv.shape)  # 打印qkv的形状
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q, k = map(F.normalize, (q, k))

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        q = q / (C // self.num_heads) ** 0.5
        k = k / (C // self.num_heads) ** 0.5

        attn = torch.einsum('bhnd,bhkd->bhnk', q, k)
        attn = attn / self.temperature
        attn = attn.softmax(dim=-1)

        if mask is not None:
            attn = attn * mask

        attn = self.attn_drop(attn)

        out = torch.einsum('bhnk,bhkd->bhnd', attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.xca = XCA(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        # 确保x的形状是(B, N, C)
        if x.dim() == 2:
            # 如果x的形状是(B, C)，则添加一个维度变为(B, 1, C)
            x = x.unsqueeze(1)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        x = torch.matmul(attn_weights, v)
        x = self.xca(x)
        x = self.output_layer(x)
        return x


class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(FullyConnected, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        ]
        for i in range(num_layers - 2):
            # 添加自注意力层
            net.extend([
                SelfAttentionBlock(hidden_size, hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, output_size),
        ])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)



