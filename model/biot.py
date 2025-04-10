import time
import math
import ninja
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
        mlstm = True,
        slstm = True,
        use_full_sample = False,
        full_sample_method = "attention",
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mlstm = mlstm
        self.slstm = slstm
        self.use_full_sample = use_full_sample
        self.full_sample_method = full_sample_method

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        if not self.mlstm and not self.slstm:
            self.transformer = LinearAttentionTransformer(
                dim=emb_size,
                heads=heads,
                depth=depth,
                max_seq_len=1024,
                attn_layer_dropout=0.2,  # dropout right after self-attention layer
                attn_dropout=0.2,  # dropout post-attention
            )

        if self.use_full_sample:
            # TODO: Implement full sample processing
            pass

        if self.mlstm:
            m_cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        num_heads=heads,
                        dropout=0.2,
                    )
                ),
                embedding_dim=emb_size,
                context_length=1024,
                num_blocks=depth
            )

            self.x_m_lstm_stack = xLSTMBlockStack(m_cfg)
        
        if self.slstm:
            s_cfg = xLSTMBlockStackConfig(
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="cuda",
                        embedding_dim=emb_size,
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
                context_length=256,
                num_blocks=1
            )

            self.x_s_lstm_stack = xLSTMBlockStack(s_cfg)

        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        if not self.use_full_sample:
            # Original approach - process each channel separately
            emb_seq = []
            for i in range(x.shape[1]):
                channel_spec_emb = self.stft(x[:, i : i + 1, :])
                channel_spec_emb = self.patch_embedding(channel_spec_emb)
                batch_size, ts, _ = channel_spec_emb.shape
                # (batch_size, ts, emb)
                channel_token_emb = (
                    self.channel_tokens(self.index[i + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )
                # (batch_size, ts, emb)
                channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

                # perturb
                if perturb:
                    ts = channel_emb.shape[1]
                    ts_new = np.random.randint(ts // 2, ts)
                    selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                    channel_emb = channel_emb[:, selected_ts]
                emb_seq.append(channel_emb)

            # (batch_size, 16 * ts, emb)
            emb = torch.cat(emb_seq, dim=1)
        else:
            # Process the entire sample as a single token
            batch_size = x.shape[0]
            # Create a combined embedding for all channels
            combined_embedding = None
            
            if self.full_sample_method == "attention":
                # Approach 3: Process each channel and use attention to combine
                emb_seq = []
                for i in range(x.shape[1]):
                    channel_spec_emb = self.stft(x[:, i : i + 1, :])
                    channel_spec_emb = self.patch_embedding(channel_spec_emb)
                    
                    # Add channel tokens
                    batch_size, ts, _ = channel_spec_emb.shape
                    channel_token_emb = (
                        self.channel_tokens(self.index[i + n_channel_offset])
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(batch_size, ts, 1)
                    )
                    channel_emb = channel_spec_emb + channel_token_emb
                    emb_seq.append(channel_emb)
                
                # Stack all channel embeddings: [batch, channels, time, emb]
                stacked_embs = torch.stack(emb_seq, dim=1)
                
                # Create/use attention mechanism if it doesn't exist
                if not hasattr(self, 'channel_attention'):
                    self.channel_attention = nn.MultiheadAttention(
                        embed_dim=stacked_embs.shape[-1],
                        num_heads=min(8, stacked_embs.shape[-1] // 32),
                        batch_first=True
                    ).to(x.device)
                
                # Reshape to [batch*time, channels, emb] for attention
                batch_size, n_channels, ts, emb_dim = stacked_embs.shape
                reshaped_embs = stacked_embs.reshape(batch_size*ts, n_channels, emb_dim)
                
                # Create a learnable query token if it doesn't exist
                if not hasattr(self, 'query_token'):
                    self.query_token = nn.Parameter(torch.randn(1, 1, emb_dim)).to(x.device)
                
                # Expand query token to batch size
                query = self.query_token.expand(batch_size*ts, 1, emb_dim)
                
                # Apply attention to get a single token per timestamp
                attended_embs, _ = self.channel_attention(query, reshaped_embs, reshaped_embs)
                
                # Reshape back to [batch, time, emb]
                combined_embedding = attended_embs.reshape(batch_size, ts, emb_dim)

            if self.full_sample_method == "convolution":
                # Process each channel and use convolution to combine
                emb_seq = []
                for i in range(x.shape[1]):
                    # Apply frequency embedding to each channel
                    channel_spec_emb = self.stft(x[:, i : i + 1, :])
                    # Apply patch embedding to each channel
                    channel_spec_emb = self.patch_embedding(channel_spec_emb)
                    emb_seq.append(channel_spec_emb)
                
                # Stack all channel embeddings: [batch, channels, time, emb]
                stacked_embs = torch.stack(emb_seq, dim=1)
                batch_size, n_channels, ts, emb_dim = stacked_embs.shape
                
                # Create channel convolution if it doesn't exist
                if not hasattr(self, 'channel_conv'):
                    self.channel_conv = nn.Sequential(
                        nn.Conv1d(n_channels, 1, kernel_size=3, padding=1),
                        nn.ELU()
                    ).to(x.device)
                
                # Reshape for convolution: [batch*ts, channels, emb]
                # Allow to reduce of channels by 
                reshaped_embs = stacked_embs.reshape(batch_size*ts, n_channels, emb_dim)
                
                # Apply convolution across channels
                conv_embs = self.channel_conv(reshaped_embs)  # shape: [batch*ts, 1, emb]
                
                # Reshape back to [batch, time, emb]
                combined_embedding = conv_embs.reshape(batch_size, ts, emb_dim)
            
            # Apply positional encoding
            # Provide postional information (order) of the time steps
            emb = self.positional_encoding(combined_embedding)
        
        # Continue with existing processing
        if self.mlstm:
            emb = self.x_m_lstm_stack(emb)
        if self.slstm:
            emb = self.x_s_lstm_stack(emb)
        if not self.slstm and not self.mlstm:
            emb = self.transformer(emb)
            
        return emb.mean(dim=1)


# supervised classifier module
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, mlstm=True, slstm=True, 
                 use_full_sample=False, full_sample_method="attention", **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, 
                               mlstm=mlstm, slstm=slstm, 
                               use_full_sample=use_full_sample, 
                               full_sample_method=full_sample_method, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x


# unsupervised pre-train module
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, **kwargs):
        super(UnsupervisedPretrain, self).__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x, n_channel_offset=0):
        emb = self.biot(x, n_channel_offset, perturb=True)
        emb = self.prediction(emb)
        pred_emb = self.biot(x, n_channel_offset)
        return emb, pred_emb


# supervised pre-train module
class SupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, mlstm=True, slstm=True, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, mlstm=mlstm, slstm=slstm)
        self.classifier_chb_mit = ClassificationHead(emb_size, 1)
        self.classifier_iiic_seizure = ClassificationHead(emb_size, 6)
        self.classifier_tuab = ClassificationHead(emb_size, 1)
        self.classifier_tuev = ClassificationHead(emb_size, 6)

    def forward(self, x, task="chb-mit"):
        x = self.biot(x)
        if task == "chb-mit":
            x = self.classifier_chb_mit(x)
        elif task == "iiic-seizure":
            x = self.classifier_iiic_seizure(x)
        elif task == "tuab":
            x = self.classifier_tuab(x)
        elif task == "tuev":
            x = self.classifier_tuev(x)
        else:
            raise NotImplementedError
        return x


if __name__ == "__main__":
    x = torch.randn(16, 2, 2000)
    model = BIOTClassifier(n_fft=200, hop_length=200, depth=4, heads=8)
    out = model(x)
    print(out.shape)

    model = UnsupervisedPretrain(n_fft=200, hop_length=200, depth=4, heads=8)
    out1, out2 = model(x)
    print(out1.shape, out2.shape)
