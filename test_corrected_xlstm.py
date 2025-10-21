"""
Fixed xLSTM configuration based on the paper specifications
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_correct_xlstm_configs(emb_size=256, heads=8, depth=4, context_length=1024):
    """
    Create correct xLSTM configurations based on the paper
    """
    
    print("Creating CORRECT xLSTM configurations...")
    
    # 1. mLSTM only configuration - xLSTM[1:0]
    mlstm_cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                dropout=0.2,
            )
        ),
        slstm_block=None,  # Explicitly set to None
        embedding_dim=emb_size,
        context_length=context_length,
        num_blocks=depth,
    )
    
    # 2. sLSTM only configuration - xLSTM[0:1] 
    slstm_cfg = xLSTMBlockStackConfig(
        mlstm_block=None,  # Explicitly set to None
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
    
    # 3. Mixed configuration - xLSTM[3:1] (3 mLSTM blocks + 1 sLSTM block)
    # This should create a different architecture with interleaved blocks
    mixed_cfg = xLSTMBlockStackConfig(
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
        slstm_at=[1]
    )
    
    return mlstm_cfg, slstm_cfg, mixed_cfg

if __name__ == "__main__":
    emb_size = 256
    heads = 8
    depth = 8
    context_length = 1024
    
    print("=" * 80)
    print("CORRECTED xLSTM CONFIGURATIONS")
    print("=" * 80)
    
    mlstm_cfg, slstm_cfg, mixed_cfg = create_correct_xlstm_configs(
        emb_size, heads, depth, context_length
    )
    
    print(f"\nTesting corrected configurations...")
    
    try:
        # Create models
        mlstm_stack = xLSTMBlockStack(mlstm_cfg)
        mlstm_params = count_parameters(mlstm_stack)
        print(f"✅ mLSTM only: {mlstm_params:,} parameters")
        
        slstm_stack = xLSTMBlockStack(slstm_cfg) 
        slstm_params = count_parameters(slstm_stack)
        print(f"✅ sLSTM only: {slstm_params:,} parameters")
        
        try:
            mixed_stack = xLSTMBlockStack(mixed_cfg)
            mixed_params = count_parameters(mixed_stack)
            print(f"✅ Mixed s+mLSTM: {mixed_params:,} parameters")
        except Exception as e:
            print(f"❌ Mixed configuration failed: {e}")
            print("Trying alternative mixed configuration...")
            # Alternative: just use both block configs without block_idx
            alt_mixed_cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(dropout=0.2)
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
            mixed_stack = xLSTMBlockStack(alt_mixed_cfg)
            mixed_params = count_parameters(mixed_stack)
            print(f"✅ Alternative Mixed s+mLSTM: {mixed_params:,} parameters")
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    print(f"\nParameter comparison:")
    print(f"  mLSTM only:     {mlstm_params:,}")
    print(f"  sLSTM only:     {slstm_params:,}")
    try:
        print(f"  Mixed s+mLSTM:  {mixed_params:,}")
        if mixed_params == mlstm_params:
            print(f"  ⚠️  WARNING: Mixed config has same params as mLSTM only!")
        else:
            print(f"  ✅ Mixed config has different parameter count")
    except:
        print(f"  ❌ Mixed config failed to create")