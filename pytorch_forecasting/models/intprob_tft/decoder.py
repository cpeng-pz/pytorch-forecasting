import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, 
                 pred_len=0, case=0,
                 d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # case 0: self-attention input_len = label_len + pred_len, cross-attention input_len = label_len + pred_len
        # case 1: self-attention input_len = label_len + pred_len, cross-attention input_len = pred_len
        # case 2: self-attention input_len = pred_len, cross-attention input_len = pred_len
        self.case = case
        self.pred_len = pred_len

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        if self.case == 0 or 1:
            x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
            )[0])
        else:
            x = x[:,-self.pred_len:,:] + self.dropout(self.self_attention(
                x[:,-self.pred_len:,:], x[:,-self.pred_len:,:], x[:,-self.pred_len:,:],
                attn_mask=x_mask
            )[0])
        x = self.norm1(x)

        if self.case == 0:  
            x = x + self.dropout(self.cross_attention(
                x, cross, cross,
                attn_mask=cross_mask
            )[0])
        else:
            x = x[:,-self.pred_len:,:] + self.dropout(self.cross_attention(
                x[:,-self.pred_len:,:], cross, cross,
                attn_mask=cross_mask
            )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x