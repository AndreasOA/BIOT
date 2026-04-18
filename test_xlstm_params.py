#!/usr/bin/env python3
"""
Test script to verify xLSTM parameter counts and configurations
"""
import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from linear_attention_transformer import LinearAttentionTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model, name):
    params = count_parameters(model)
    print(f"{name}: {params:,} parameters")
    return params

# Configuration parameters
emb_size = 256
heads = 8
depth = 4
context_length = 1024

print("=" * 80)
print("xLSTM PARAMETER COUNT VERIFICATION")
print("=" * 80)

print(f"\nConfiguration:")
print(f"  Embedding size: {emb_size}")
print(f"  Heads: {heads}")
print(f"  Depth: {depth}")
print(f"  Context length: {context_length}")

print(f"\n" + "-" * 60)
print("1. ORIGINAL CONFIGURATIONS (as in your code)")
print("-" * 60)

# 1. mLSTM only (as in your current code)
print("\n1a. mLSTM Only Configuration:")
m_cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            dropout=0.2,
        )
    ),
    embedding_dim=emb_size,
    context_length=context_length,
    num_blocks=depth
)
m_lstm_stack = xLSTMBlockStack(m_cfg)
m_params = print_model_info(m_lstm_stack, "mLSTM stack")

# 2. sLSTM only (as in your current code)
print("\n1b. sLSTM Only Configuration:")
s_cfg = xLSTMBlockStackConfig(
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            embedding_dim=emb_size,
            num_heads=heads,
            dropout=0.2,
        ),
        feedforward=FeedForwardConfig(
            proj_factor=1.3,
            act_fn="gelu",
            embedding_dim=emb_size,
            dropout=0.0,
            bias=False,
            ff_type="ffn_gated"
        ),
    ),
    embedding_dim=emb_size,
    context_length=context_length,
    num_blocks=depth
)
s_lstm_stack = xLSTMBlockStack(s_cfg)
s_params = print_model_info(s_lstm_stack, "sLSTM stack")

# 3. Mixed configuration (as in your current code)
print("\n1c. Mixed s+mLSTM Configuration:")
sm_cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            dropout=0.2,
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            embedding_dim=emb_size,
            num_heads=heads,
            dropout=0.2,
        ),
        feedforward=FeedForwardConfig(
            proj_factor=1.3,
            act_fn="gelu",
            embedding_dim=emb_size,
            dropout=0.0,
            bias=False,
            ff_type="ffn_gated"
        ),
    ),
    embedding_dim=emb_size,
    context_length=context_length,
    num_blocks=depth,
)
sm_lstm_stack = xLSTMBlockStack(sm_cfg)
sm_params = print_model_info(sm_lstm_stack, "Mixed s+mLSTM stack")

print(f"\n" + "-" * 60)
print("2. LINEAR TRANSFORMER FOR COMPARISON")
print("-" * 60)

# Linear Transformer
print("\n2. Linear Attention Transformer:")
try:
    transformer = LinearAttentionTransformer(
        dim=emb_size,
        heads=heads,
        depth=depth,
        max_seq_len=1024,
        attn_layer_dropout=0.2,
        attn_dropout=0.2,
    )
    t_params = print_model_info(transformer, "Linear Transformer")
except Exception as e:
    print(f"Error creating Linear Transformer: {e}")
    t_params = 0

print(f"\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

print(f"\nParameter counts:")
print(f"  mLSTM only:     {m_params:,}")
print(f"  sLSTM only:     {s_params:,}")
print(f"  s+mLSTM mixed:  {sm_params:,}")
if t_params > 0:
    print(f"  Linear Trans:   {t_params:,}")

# Check if all xLSTM variants have the same parameter count
xlstm_counts = [m_params, s_params, sm_params]
if len(set(xlstm_counts)) == 1:
    print(f"\n❌ PROBLEM DETECTED:")
    print(f"   All xLSTM variants have identical parameter counts: {xlstm_counts[0]:,}")
    print(f"   This indicates the configuration is not working as expected.")
    
    # Check the configuration details
    print(f"\n🔍 INVESTIGATING CONFIGURATIONS:")
    print(f"   mLSTM config has slstm_block: {hasattr(m_cfg, 'slstm_block') and m_cfg.slstm_block is not None}")
    print(f"   sLSTM config has mlstm_block: {hasattr(s_cfg, 'mlstm_block') and s_cfg.mlstm_block is not None}")
    print(f"   Mixed config has both blocks: {hasattr(sm_cfg, 'mlstm_block') and sm_cfg.mlstm_block is not None and hasattr(sm_cfg, 'slstm_block') and sm_cfg.slstm_block is not None}")
    
else:
    print(f"\n✅ Parameter counts differ as expected")

print(f"\n" + "-" * 60)
print("3. CHECKING MODEL ARCHITECTURES")
print("-" * 60)

# Print model architectures to understand what's being created
print(f"\nmLSTM model structure:")
for i, (name, param) in enumerate(m_lstm_stack.named_parameters()):
    if i < 5:  # Show first 5 parameters
        print(f"  {name}: {param.shape}")
    elif i == 5:
        print(f"  ... ({sum(1 for _ in m_lstm_stack.named_parameters())} total parameters)")
        break

print(f"\nsLSTM model structure:")
for i, (name, param) in enumerate(s_lstm_stack.named_parameters()):
    if i < 5:  # Show first 5 parameters
        print(f"  {name}: {param.shape}")
    elif i == 5:
        print(f"  ... ({sum(1 for _ in s_lstm_stack.named_parameters())} total parameters)")
        break

print(f"\nMixed s+mLSTM model structure:")
for i, (name, param) in enumerate(sm_lstm_stack.named_parameters()):
    if i < 5:  # Show first 5 parameters
        print(f"  {name}: {param.shape}")
    elif i == 5:
        print(f"  ... ({sum(1 for _ in sm_lstm_stack.named_parameters())} total parameters)")
        break