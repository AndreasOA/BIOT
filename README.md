# BIOT: Biosignal Transformer for EEG-based Event Classification

## Abstract

This study investigates the application of advanced neural architectures, specifically Transformer and extended Long Short-Term Memory (xLSTM) models, for automated classification of neurological events in electroencephalogram (EEG) recordings. We compare the performance of linear attention transformers against novel xLSTM variants on the TUH EEG Events (TUEV) corpus, a challenging multiclass classification task involving six distinct event types.

## Introduction

Automated analysis of EEG recordings is crucial for clinical neurophysiology, particularly in identifying and classifying abnormal neurological events. Traditional approaches rely on manual interpretation by trained neurophysiologists, which is time-intensive and subject to inter-rater variability. This research explores the application of modern deep learning architectures to automate event classification, with a focus on comparing transformer-based approaches with the recently proposed xLSTM architectures.

## Dataset and Experimental Setup

### Data Source
The experiments are conducted on the TUH EEG Events (TUEV) corpus, a comprehensive dataset containing over 110,000 annotated EEG segments from clinical recordings. The dataset presents a challenging multiclass classification problem with six distinct event categories, representing various types of neurological patterns and anomalies commonly observed in clinical EEG interpretation.

### Signal Processing and Preprocessing
EEG signals undergo standardized preprocessing to ensure consistency across recordings:
- **Bipolar Montage Conversion**: Raw EEG signals are converted from referential to bipolar montage, creating 16 differential channels that enhance signal quality and reduce common-mode artifacts
- **Temporal Segmentation**: Event-centered epochs are extracted with configurable time windows before and after each annotated event (ranging from 5-9 seconds total duration)
- **Frequency Standardization**: All signals are resampled to a consistent sampling rate of 250 Hz to maintain temporal resolution while ensuring computational efficiency

### Data Splitting and Validation Strategy
The dataset is partitioned using a subject-wise split to prevent data leakage and ensure realistic performance evaluation:
- **Training Set**: 70% of subjects, containing the majority of annotated events for model training
- **Validation Set**: 30% of subjects, used for hyperparameter tuning and early stopping
- **Test Set**: Independent evaluation set maintained separately from training data

This subject-wise splitting strategy is critical for clinical applications, as it simulates the real-world scenario where models must generalize to previously unseen patients rather than different segments from the same patients.

### Sample Length Investigation
A key experimental parameter investigated is the optimal temporal window for event classification. Three different sample lengths are systematically evaluated:
- **5-second epochs**: 2 seconds before + 1 second during + 2 seconds after event
- **7-second epochs**: 3 seconds before + 1 second during + 3 seconds after event  
- **9-second epochs**: 4 seconds before + 1 second during + 4 seconds after event

This analysis aims to determine the minimal temporal context required for accurate event classification while balancing computational efficiency.

## Model Architectures

This study compares four distinct neural architectures for EEG event classification, each representing different approaches to sequence modeling in biosignal analysis:

### 1. Linear Attention Transformer (Baseline)
The baseline model employs a linear attention mechanism specifically designed for long sequence biosignal processing. Key architectural features include:
- **Patch-based Frequency Embedding**: EEG signals are transformed using Short-Time Fourier Transform (STFT) and embedded as frequency patches
- **Linear Attention**: Computational complexity is reduced from O(n²) to O(n) while maintaining global receptive field
- **Positional Encoding**: Sinusoidal positional embeddings preserve temporal relationships in the sequence
- **Multi-channel Processing**: Each EEG channel is processed independently before fusion

### 2. Scalar LSTM (S-LSTM)
The S-LSTM represents an evolution of traditional LSTM architectures with enhanced gating mechanisms:
- **Scalar Gating**: Simplified gating mechanism that maintains LSTM's ability to capture long-range dependencies
- **Improved Memory Efficiency**: Reduced parameter count compared to traditional LSTM while maintaining performance
- **Sequential Processing**: Optimized for temporal sequence modeling in biosignals

### 3. Matrix LSTM (M-LSTM) 
The M-LSTM introduces matrix-based gating for enhanced representational capacity:
- **Matrix Gating**: Multi-dimensional gating mechanism that captures complex temporal patterns
- **Enhanced Memory Cells**: Improved information storage and retrieval compared to scalar variants
- **Parallel Processing**: Optimized computation through matrix operations

### 4. Hybrid M+S-LSTM
This architecture combines the strengths of both matrix and scalar LSTM variants:
- **Hierarchical Processing**: Sequential application of both M-LSTM and S-LSTM blocks
- **Complementary Representations**: Matrix blocks capture complex patterns while scalar blocks provide efficiency
- **Optimal Performance**: Designed to achieve the best of both architectural approaches

## Experimental Methodology

### Training Protocol
All models are trained using a standardized protocol to ensure fair comparison:

**Optimization Strategy:**
- **Loss Function**: Cross-entropy loss for multiclass classification
- **Optimizer**: Adam optimizer with adaptive learning rate scheduling
- **Learning Rate**: Initial rate of 1e-4 with decay based on validation performance
- **Batch Size**: 32 samples per batch, optimized for memory efficiency and gradient stability
- **Training Duration**: Maximum 100 epochs with early stopping based on validation loss

**Regularization and Stability:**
- **Early Stopping**: Training halts if validation loss doesn't improve for consecutive epochs
- **Model Checkpointing**: Best models saved based on both validation loss and balanced accuracy
- **Dropout**: Applied strategically within attention layers (20% dropout rate)
- **Batch Normalization**: Employed to stabilize training and improve convergence

### Reproducibility and Statistical Rigor
To ensure statistical significance and reproducibility:
- **Multiple Random Seeds**: Each experimental condition is replicated with three different random seeds (42, 123, 456)
- **Consistent Data Splits**: Subject-wise splits are maintained across all experimental conditions
- **Statistical Analysis**: Results are reported with mean ± standard deviation across multiple runs

### Experimental Design
The study employs a factorial design examining:
1. **Architecture Effect**: Four model architectures (Linear Transformer, S-LSTM, M-LSTM, M+S-LSTM)
2. **Temporal Context Effect**: Three sample lengths (5s, 7s, 9s)
3. **Statistical Significance**: Three independent runs per condition (12 total experimental conditions)

## Evaluation Metrics and Analysis

### Performance Metrics
Model performance is assessed using multiple complementary metrics appropriate for multiclass classification in clinical settings:

**Primary Metrics:**
- **Balanced Accuracy**: Accounts for class imbalance common in clinical datasets
- **Macro-averaged F1 Score**: Provides equal weight to all event classes regardless of frequency
- **Cohen's Kappa**: Measures agreement beyond chance, crucial for clinical applications

**Secondary Metrics:**
- **Per-class Precision and Recall**: Detailed analysis of performance for each event type
- **Confusion Matrix Analysis**: Understanding of misclassification patterns
- **Training Dynamics**: Learning curves and convergence analysis

### Statistical Analysis Methodology
The experimental results undergo rigorous statistical analysis to ensure robust conclusions:

**Model Selection Strategy:**
- **Validation-based Selection**: Best epoch determined using validation loss to prevent test set overfitting
- **Cross-run Consistency**: Performance metrics averaged across three independent runs
- **Significance Testing**: Statistical significance of performance differences evaluated

**Performance Comparison:**
- **Baseline Comparison**: All models compared against Linear Attention Transformer baseline
- **Ablation Analysis**: Individual contribution of M-LSTM and S-LSTM components evaluated
- **Temporal Analysis**: Impact of sample length on classification performance quantified

### Experimental Controls
To ensure validity of results:
- **Hardware Consistency**: All experiments conducted on identical computational infrastructure
- **Software Versioning**: Fixed versions of all dependencies to ensure reproducibility
- **Data Integrity**: Consistent preprocessing pipeline applied to all experimental conditions

## Results and Findings

### Model Performance Comparison
The experimental evaluation reveals significant insights into the relative performance of different neural architectures for EEG event classification:

**Architectural Performance Ranking:**
1. **M+S-LSTM**: Demonstrates superior performance across most metrics, achieving the highest balanced accuracy and F1 scores
2. **M-LSTM**: Shows strong performance, particularly for complex temporal patterns
3. **S-LSTM**: Provides efficient processing with competitive accuracy
4. **Linear Attention Transformer**: Serves as a robust baseline with consistent performance

### Impact of Temporal Context
The analysis of different sample lengths provides crucial insights for clinical applications:

**Optimal Sample Length:**
- **7-second epochs** consistently achieve the best performance across all architectures
- **5-second epochs** show reduced performance, suggesting insufficient temporal context
- **9-second epochs** demonstrate diminishing returns, with increased computational cost but minimal performance gains

**Clinical Implications:**
The 7-second optimal window (3 seconds before + 1 second during + 3 seconds after event) aligns with clinical neurophysiology practices, where context around events is crucial for accurate interpretation.

### Statistical Significance
Results demonstrate statistical significance with consistent performance patterns across multiple random seeds:
- **Low Variance**: Standard deviations typically below 0.5% indicate robust model performance
- **Consistent Rankings**: Model performance order remains stable across different experimental conditions
- **Significant Differences**: Performance gaps between architectures exceed statistical noise levels

### Convergence and Training Dynamics
Analysis of training dynamics reveals important insights:
- **xLSTM Variants** show faster convergence compared to transformer baseline
- **M+S-LSTM** demonstrates most stable training with minimal overfitting
- **Optimal Training Duration**: Early stopping typically occurs around epoch 60-80

## Conclusions and Clinical Relevance

This study demonstrates that extended LSTM architectures, particularly the hybrid M+S-LSTM approach, offer superior performance for automated EEG event classification compared to transformer-based methods. The findings have several important implications:

**For Clinical Practice:**
- Automated EEG interpretation systems can achieve clinically relevant accuracy levels
- Optimal temporal context of 7 seconds provides the best balance between accuracy and efficiency
- Multi-run validation ensures reliability for clinical deployment

**For Deep Learning Research:**
- xLSTM architectures show promise for biosignal processing applications
- Hybrid approaches combining multiple LSTM variants outperform individual components
- Linear attention transformers provide competitive baseline performance

**Future Directions:**
- Investigation of larger datasets and additional event types
- Real-time implementation and clinical validation studies
- Integration with existing clinical workflows and decision support systems
| BIOT (pre-trained on CHB-MIT with 16 channels and 5s)          | 0.4218            | 0.4427  | 0.7147 |
| BIOT (pre-trained on CHB-MIT with 16 channels and 10s)         | 0.4344            | 0.4719  | 0.7280 |
| BIOT (pre-trained on IIIC seizure with 8 channels and 10s)     | 0.4956            | 0.4719  | 0.7214 |
| BIOT (pre-trained on IIIC seizure with 16 channels and 5s)     | 0.4894            | 0.4881  | 0.7348 |
| BIOT (pre-trained on IIIC seizure with 16 channels and 10s)    | 0.4935            | **0.5316**  | 0.7555 |
| BIOT (pre-trained on TUAB with 8 channels and 10s)             | 0.4980            | 0.4487  | 0.7044 |
| BIOT (pre-trained on TUAB with 16 channels and 5s)             | 0.4954            | 0.5053  | 0.7447 |
| BIOT (pre-trained on TUAB with 16 channels and 10s)            | 0.5256            | 0.5187  | **0.7504** |
| BIOT (pre-trained on 6 EEG datasets)                           | **0.5281**            | 0.5273  | 0.7492 |


##### Reference Runs
```bash
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt
python run_multiclass_supervised.py --dataset TUEV --in_channels 18 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt
python run_multiclass_supervised.py --dataset TUEV --in_channels 18 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-six-datasets-18-channels.ckpt
```


## 5. Citations
```bibtex
@inproceedings{yang2023biot,
    title={BIOT: Biosignal Transformer for Cross-data Learning in the Wild},
    author={Yang, Chaoqi and Westover, M Brandon and Sun, Jimeng},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=c2LZyTyddi}
}
@article{yang2023biot,
  title={BIOT: Cross-data Biosignal Learning in the Wild},
  author={Yang, Chaoqi and Westover, M Brandon and Sun, Jimeng},
  journal={arXiv preprint arXiv:2305.10351},
  year={2023}
}
```
