# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
# flash-attention 
import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.functional import linear

from einops import rearrange
from mmcv.runner import auto_fp16
from mmcv.cnn import constant_init

from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input
import torch.nn.functional as F

def _in_projection_packed(q, k, v, w, b = None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D) 
            kv: The tensor containing the key, and value. (B, S, 2, H, D) 
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k 
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=kv.device)                    
            output = flash_attn_varlen_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_varlen_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


class MultiScaleDeformableAttnWithQuery(nn.Module):
    """Multi-scale deformable attention with learnable Q/K projections.
    
    This module implements deformable attention where attention weights are computed
    via query-key similarity (like standard Transformer), rather than being directly
    predicted from the query.
    
    Args:
        embed_dims (int): The embedding dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_levels (int): Number of feature pyramid levels. Default: 4.
        num_points (int): Number of sampling points per query. Default: 4.
        bias (bool): Whether to use bias in projections. Default: True.
    """
    
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=13, bias=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.fp16_enabled = True
        
        # Check that embed_dims is divisible by num_heads
        assert embed_dims % num_heads == 0, \
            f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
        
        self.head_dims = embed_dims // num_heads
        
        # Query and Key projection layers (like standard Transformer)
        self.query_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.key_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize projection weights using Xavier initialization."""
        constant_init(self.query_proj, val=0.0, bias=0.0)
        constant_init(self.key_proj, val=0.0, bias=0.0)
    
    def forward(self, query, value, value_spatial_shapes, sampling_locations):
        """Forward pass with Q/K projections.
        
        Args:
            query (torch.Tensor): Query tensor of shape
                (bs, num_queries, num_heads, head_dims)
            value (torch.Tensor): Value tensor of shape
                (bs, num_keys, num_heads, head_dims)
            value_spatial_shapes (torch.Tensor): Spatial shapes of shape
                (num_levels, 2)
            sampling_locations (torch.Tensor): Sampling locations of shape
                (bs, num_queries, num_heads, num_levels, num_points, 2)
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: (bs, num_queries, embed_dims)
                - attention_weights: (bs, num_queries, num_heads, num_levels*num_points)
        """
        bs, num_queries, num_heads, head_dims = query.shape
        
        # Reshape for projection: [bs, num_queries, num_heads, head_dims] -> [bs, num_queries, embed_dims]
        query_flat = query.flatten(2)
        value_flat = value.flatten(2)
        
        # Apply Q/K projections
        query_proj = self.query_proj(query_flat)  # [bs, num_queries, embed_dims]
        value_proj = self.key_proj(value_flat)    # [bs, num_keys, embed_dims]
        
        # Reshape back to multi-head format
        query_proj = query_proj.reshape(bs, num_queries, num_heads, head_dims)
        value_proj = value_proj.reshape(bs, -1, num_heads, head_dims)
        
        # Call the functional implementation
        return self.multi_scale_deformable_attn_with_query(
            query_proj, value_proj, value_spatial_shapes, sampling_locations, use_flash=False
        )

    def multi_scale_deformable_attn_with_query(
        self, query: torch.Tensor, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor, use_flash: bool = True) -> torch.Tensor:
        """Multi-scale deformable attention with Flash Attention support.
        
        Uses flash_attn for efficient computation when use_flash=True and inputs are on CUDA
        with appropriate dtype (fp16/bf16).

        Args:
            query (torch.Tensor): The query has shape
                (bs, num_queries, num_heads, embed_dims//num_heads)
            value (torch.Tensor): The value has shape
                (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs, num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            use_flash (bool): Whether to use Flash Attention. Default: True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - output: has shape (bs, num_queries, embed_dims)
                - attention_weights: has shape (bs, num_queries, num_heads, num_levels*num_points)
        """
        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
        
        # Sample values from multi-scale feature maps
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        
        for level, (H_, W_) in enumerate(value_spatial_shapes):
            # bs, H_*W_, num_heads, embed_dims ->
            # bs, H_*W_, num_heads*embed_dims ->
            # bs, num_heads*embed_dims, H_*W_ ->
            # bs*num_heads, embed_dims, H_, W_
            value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_)
            # bs, num_queries, num_heads, num_points, 2 ->
            # bs, num_heads, num_queries, num_points, 2 ->
            # bs*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            # bs*num_heads, embed_dims, num_queries, num_points
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        
        # Stack sampled values: [bs*num_heads, embed_dims, num_queries, num_levels, num_points]
        sampled_values = torch.stack(sampling_value_list, dim=-2)
        # Reshape to: [bs*num_heads, embed_dims, num_queries, num_levels*num_points]
        sampled_values = sampled_values.flatten(-2)
        
        # Check if we can use Flash Attention
        can_use_flash = (
            use_flash and 
            query.is_cuda and 
            value.is_cuda 
        )
        
        if can_use_flash:
            # Use Flash Attention for efficient computation
            output, attention_weights_output = self._compute_attention_flash(
                query, sampled_values, bs, num_queries, num_heads, embed_dims, num_levels, num_points
            )
        else:
            # Fallback to standard attention computation
            output, attention_weights_output = self._compute_attention_standard(
                query, sampled_values, bs, num_queries, num_heads, embed_dims, num_levels, num_points
            )
        
        return output, attention_weights_output

    @auto_fp16(apply_to=('query', 'sampled_values'), out_fp32=True)
    def _compute_attention_flash(self, query, sampled_values, bs, num_queries, num_heads, embed_dims, num_levels, num_points):
        """Compute attention using Flash Attention.
        
        Note: Flash Attention is optimized for standard attention patterns.
        For deformable attention with per-query sampling, the structure doesn't map
        perfectly to flash_attn's expectations. This implementation uses a workaround
        that treats each query independently.
        """
        # print("Using Flash Attention for deformable attention computation.")
        # try:
        from flash_attn.flash_attn_interface import flash_attn_func
        
        # sampled_values: [bs*num_heads, embed_dims, num_queries, num_levels*num_points]
        # Reshape to: [bs, num_heads, embed_dims, num_queries, num_levels*num_points]
        sampled_values = sampled_values.view(bs, num_heads, embed_dims, num_queries, num_levels * num_points)
        
        # Transpose to: [bs, num_queries, num_levels*num_points, num_heads, embed_dims]
        sampled_values = sampled_values.permute(0, 3, 4, 1, 2).contiguous()

        query = query.contiguous()
        
        # query: [bs, num_queries, num_heads, embed_dims]
        # We use sampled_values as both K and V
        
        # For flash_attn_func:
        # q: [batch_size, seqlen_q, nheads, headdim]
        # k: [batch_size, seqlen_k, nheads, headdim]
        # v: [batch_size, seqlen_k, nheads, headdim]
        
        # In our case:
        # q: [bs, num_queries, num_heads, embed_dims]
        # k, v: [bs, num_queries, num_levels*num_points, num_heads, embed_dims]
        
        # However, flash_attn expects all queries to attend to the same KV
        # In our case, each query has different sampled points
        # This doesn't fit flash_attn's paradigm well
        
        # Workaround: Process each batch sample independently
        outputs = []
        # attn_weights_list = []
        
        for b in range(bs):
            # For this batch: treat each query as a separate "batch" in flash attn
            q_b = query[b:b+1]  # [1, num_queries, num_heads, embed_dims]
            kv_b = sampled_values[b:b+1]  # [1, num_queries, num_levels*num_points, num_heads, embed_dims]
            
            # Reshape: [num_queries, 1, num_heads, embed_dims]
            q_b = q_b.transpose(0, 1)  # [num_queries, 1, num_heads, embed_dims]
            kv_b = kv_b.squeeze(0)  # [num_queries, num_levels*num_points, num_heads, embed_dims]
            
            # Use flash_attn_func for each query
            # This is still not ideal but better than nothing
            # flash_attn_func(q, k, v, ...)
            out_b = flash_attn_func(
                q_b, kv_b, kv_b,  # q, k, v
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(embed_dims),
                causal=False,
                return_attn_probs=False  # We'll compute weights separately if needed
            )  # [num_queries, 1, num_heads, embed_dims]
            
            out_b = out_b.transpose(0, 1)  # [1, num_queries, num_heads, embed_dims]
            outputs.append(out_b)
        
        # Concatenate batch results
        output = torch.cat(outputs, dim=0)  # [bs, num_queries, num_heads, embed_dims]
        output = output.flatten(2)  # [bs, num_queries, num_heads*embed_dims]
        
        # For attention weights, we need to compute them separately
        # since flash_attn doesn't return them efficiently
        # Fall back to standard computation for weights
        # _, attention_weights_output = _compute_attention_standard(
        #     query.view(bs, num_queries, num_heads, embed_dims),
        #     sampled_values.permute(0, 3, 4, 1, 2).reshape(bs * num_heads, embed_dims, num_queries, num_levels * num_points),
        #     bs, num_queries, num_heads, embed_dims, num_levels, num_points
        # )
        
        return output, None

    @auto_fp16(apply_to=('query', 'sampled_values'), out_fp32=True)
    def _compute_attention_standard(self, query, sampled_values, bs, num_queries, num_heads, embed_dims, num_levels, num_points):
        """Compute attention using standard PyTorch operations."""
        # Prepare query for similarity computation
        # query: [bs, num_queries, num_heads, embed_dims] ->
        # [bs, num_heads, num_queries, embed_dims] ->
        # [bs*num_heads, num_queries, embed_dims]
        # print("Using standard attention computation.")
        query_reshaped = query.transpose(1, 2).reshape(bs * num_heads, num_queries, embed_dims)
        
        # sampled_values: [bs*num_heads, embed_dims, num_queries, num_levels*num_points]
        # Transpose sampled_values: [bs*num_heads, num_queries, num_levels*num_points, embed_dims]
        sampled_values_t = sampled_values.permute(0, 2, 3, 1)
        
        # Compute scaled dot-product attention weights
        scale = 1.0 / math.sqrt(embed_dims)
        attention_scores = torch.einsum('bqd,bqpd->bqp', query_reshaped, sampled_values_t) * scale
        
        # Apply softmax to get attention weights
        # [bs*num_heads, num_queries, num_levels*num_points]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Reshape for weighted sum: [bs*num_heads, 1, num_queries, num_levels*num_points]
        attention_weights_reshaped = attention_weights.unsqueeze(1)
        
        # Compute weighted output
        # sampled_values: [bs*num_heads, embed_dims, num_queries, num_levels*num_points]
        # attention_weights_reshaped: [bs*num_heads, 1, num_queries, num_levels*num_points]
        # output: [bs*num_heads, embed_dims, num_queries]
        output = (sampled_values * attention_weights_reshaped).sum(-1)
        
        # Reshape output: [bs*num_heads, embed_dims, num_queries] ->
        # [bs, num_heads*embed_dims, num_queries] ->
        # [bs, num_queries, num_heads*embed_dims]
        output = output.view(bs, num_heads * embed_dims, num_queries)
        output = output.transpose(1, 2).contiguous()
        
        # Reshape attention_weights to match _get_weights_img format
        # [bs*num_heads, num_queries, num_levels*num_points] ->
        # [bs, num_heads, num_queries, num_levels*num_points] ->
        # [bs, num_queries, num_heads, num_levels*num_points]
        attention_weights_output = attention_weights.view(bs, num_heads, num_queries, -1).transpose(1, 2).contiguous()
        
        return output, attention_weights_output