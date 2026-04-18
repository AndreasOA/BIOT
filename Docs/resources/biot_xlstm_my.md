# Extracted PDF Content

Total pages: 4

---

## Page 0

**Page Dimensions:** 1700x2200 (DPI: 200)

# Evaluating XLSTM using BIOT 

Andreas Oberdammer<br>Artificial Intelligence<br>Johannes Kepler University<br>k11908776@students.jku.at


#### Abstract

This work evaluates the applicability and effectiveness of integrating the extended Long Short-Term Memory (xLSTM) architecture within the BIOT (Biosignal Transformer) model framework for biosignal data classification tasks, specifically using the TUEV dataset. The primary goal was to compare the performance of the xLSTM variants (M-LSTM, S-LSTM, and M+S-LSTM) with the original Linear Transformer-based BIOT architecture. We hypothesized that while the Linear Transformer would offer computational efficiency and simplicity, the xLSTM variants would enhance the model's ability to capture complex temporal dependencies due to their sophisticated gating mechanisms and memory structures. Our comprehensive experiments, involving multiple configurations and data sample lengths, demonstrated that the S-LSTM variant significantly outperformed both the original Linear Transformer and other xLSTM configurations, confirming our expectations about the benefits of advanced memory gating and mixing capabilities provided by the xLSTM architecture.


## 1 Experimental Setup and Analysis

### 1.1 Model Architecture

We conducted experiments using the BIOT (Biosignal Transformer) architecture (1) as our base model. The key modification in our study was replacing the Linear Transformer component of BIOT with different variants of the xLSTM architecture (2). Specifically, we implemented the following model configurations:

- Linear Transformer: The original BIOT architecture.
- M+S-LSTM: Combination of M-LSTM and S-LSTM components.
- M-LSTM: Using only the M-LSTM variant.
- S-LSTM: Using only the S-LSTM variant.


### 1.2 Original Evaluation of BIOT on TUEV

The original BIOT model was evaluated comprehensively across EEG, ECG, and human sensory datasets, including TUEV. The evaluation included supervised learning tasks such as sleep and resting EEG event classification with the TUEV dataset, using metrics like Balanced Accuracy, Cohen's Kappa, and Weighted F1 Score. The original BIOT experiments typically involved five independent runs per configuration to ensure statistical robustness and reliability (1).

---

## Page 1

**Page Dimensions:** 1700x2200 (DPI: 200)

# 1.3 Data Processing and Sample Length 

For each experiment, we processed the biosignal data with different sample lengths (5, 7, and 9 seconds). The sample window is always centered on the event of interest, with the following distribution:

- 5-second window: 2 seconds before +1 second event +2 seconds after.
- 7-second window: 3 seconds before +1 second event +3 seconds after.
- 9-second window: 4 seconds before +1 second event +4 seconds after.


### 1.4 Experimental Settings

Our experimental protocol involved the following specific hyperparameter grid and conditions:

- Learning rate (lr): 0.001
- Weight decay: $1 \times 10^{-5}$
- Batch size: 128
- Number of workers: 16
- Sampling rate: 250 Hz (resampled to 200 Hz )
- Token size: 200
- Hop length: 100
- Dataset: TUEV
- Input channels: 16
- Number of classes: 6
- Epochs: 100
- Validation ratio: 0.1

The configuration of the M-LSTM and S-LSTM variants was switched after three runs of the sweep, thus each setting was evaluated across three independent runs.

### 1.5 Differences from Original BIOT Implementation

The primary differences from the original BIOT evaluation protocol are:

- Runs per configuration: Original BIOT experiments (1) conducted five runs, whereas our setup involved three runs per configuration.
- Validation set size: In the original BIOT experiment (1), a validation set size of $20 \%$ was used, whereas our experiments used only $10 \%$. While this represents a notable difference, it is important to note that in the original setup, validation metrics were primarily used for hyperparameter tuning. In contrast, we omitted hyperparameter selection via validation metrics to ensure a more direct and fair comparison, strictly adhering to the original BIOT settings. Nonetheless, we acknowledge that using a smaller validation set and more training data is not entirely equivalent-it may influence model evaluation and selection. Therefore, it should be considered when interpreting the results.
- Architectural modifications: Replacement of the original Linear Transformer component with xLSTM variants.
- Sample length variants: Explicit testing of multiple event-centered windows (5, 7, and 9 seconds).


### 1.6 Evaluation Metrics

We employed primary metrics consistent with the BIOT paper, including:

- Balanced Accuracy

---

## Page 2

**Page Dimensions:** 1700x2200 (DPI: 200)

- Cohen's Kappa
- F1 Score

Additional secondary metrics were used for comprehensive evaluation:

- Macro and Micro AUC-PR
- AUROC (Macro OVO, Macro OVR, Weighted OVO, Weighted OVR)


# 1.7 Analysis Methods 

We conducted two analyses: Comprehensive Analysis:

- Mean of the maximum values of each run
- Mean performance across runs
- Variance in performance


## Simplified Analysis:

- Mean performance across runs
- Variance indicating consistency

This detailed evaluation approach provides a clear understanding of the stability and capability of each model configuration.

| Model | Sample <br> Length | balanced acc |  |  |  | cohen |  | f1 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  |  | Max | Mean | Max | Mean | Max | Mean |
| Linear | 5 s | $0.573 \pm 0.025$ | $0.495 \pm 0.011$ | $0.553 \pm 0.015$ | $0.467 \pm 0.009$ | $0.766 \pm 0.005$ | $0.720 \pm 0.005$ |
| Transformer | 7 s | $0.543 \pm 0.013$ | $0.469 \pm 0.008$ | $0.510 \pm 0.009$ | $0.432 \pm 0.012$ | $0.743 \pm 0.008$ | $0.699 \pm 0.008$ |
|  | 9 s | $0.537 \pm 0.023$ | $0.461 \pm 0.010$ | $0.496 \pm 0.017$ | $0.419 \pm 0.015$ | $0.738 \pm 0.012$ | $0.692 \pm 0.009$ |
| M+S-LSTM | 5 s | $0.565 \pm 0.014$ | $0.501 \pm 0.005$ | $0.529 \pm 0.016$ | $0.449 \pm 0.011$ | $0.754 \pm 0.009$ | $0.712 \pm 0.005$ |
|  | 7 s | $0.555 \pm 0.039$ | $0.472 \pm 0.025$ | $0.508 \pm 0.042$ | $0.427 \pm 0.038$ | $0.743 \pm 0.020$ | $0.701 \pm 0.021$ |
|  | 9 s | $0.539 \pm 0.014$ | $0.476 \pm 0.003$ | $0.516 \pm 0.003$ | $0.439 \pm 0.004$ | $0.749 \pm 0.002$ | $0.708 \pm 0.002$ |
| M-LSTM | 5 s | $0.564 \pm 0.008$ | $0.501 \pm 0.004$ | $0.530 \pm 0.015$ | $0.445 \pm 0.008$ | $0.755 \pm 0.008$ | $0.708 \pm 0.005$ |
|  | 7 s | $0.563 \pm 0.007$ | $0.490 \pm 0.006$ | $0.504 \pm 0.023$ | $0.429 \pm 0.013$ | $0.745 \pm 0.010$ | $0.700 \pm 0.007$ |
|  | 9 s | $0.560 \pm 0.021$ | $0.493 \pm 0.008$ | $0.518 \pm 0.010$ | $0.447 \pm 0.010$ | $0.749 \pm 0.005$ | $0.711 \pm 0.006$ |
| S-LSTM | 5 s | $0.581 \pm 0.014$ | $0.527 \pm 0.016$ | $0.581 \pm 0.020$ | $0.500 \pm 0.019$ | $\mathbf{0 . 7 8 2} \pm 0.010$ | $0.737 \pm 0.011$ |
|  | 7 s | $\mathbf{0 . 6 0 9} \pm 0.014$ | $\mathbf{0 . 5 4 4} \pm 0.006$ | $\mathbf{0 . 5 8 5} \pm 0.005$ | $\mathbf{0 . 5 0 9} \pm 0.009$ | $0.781 \pm 0.002$ | $\mathbf{0 . 7 4 1} \pm 0.005$ |
|  | 9 s | $0.587 \pm 0.009$ | $0.523 \pm 0.011$ | $0.578 \pm 0.018$ | $0.490 \pm 0.024$ | $0.777 \pm 0.010$ | $0.730 \pm 0.014$ |
| Model | Sample <br> Length | aucpr macro | aucpr micro | auroc <br> macro ovo | auroc <br> macro ovr | auroc <br> weighted ovo | auroc <br> weighted ovr |
| Linear | 5 s | $0.473 \pm 0.009$ | $0.757 \pm 0.007$ | $0.825 \pm 0.007$ | $0.868 \pm 0.006$ | $0.857 \pm 0.004$ | $0.882 \pm 0.001$ |
| Transformer | 7 s | $0.448 \pm 0.014$ | $0.732 \pm 0.019$ | $0.812 \pm 0.009$ | $0.853 \pm 0.009$ | $0.841 \pm 0.008$ | $0.873 \pm 0.005$ |
|  | 9 s | $0.440 \pm 0.006$ | $0.728 \pm 0.017$ | $0.803 \pm 0.005$ | $0.847 \pm 0.006$ | $0.835 \pm 0.006$ | $0.861 \pm 0.009$ |
| M+S-LSTM | 5 s | $0.488 \pm 0.007$ | $0.767 \pm 0.004$ | $0.803 \pm 0.011$ | $0.850 \pm 0.013$ | $0.842 \pm 0.007$ | $0.860 \pm 0.003$ |
|  | 7 s | $0.452 \pm 0.014$ | $0.754 \pm 0.011$ | $0.791 \pm 0.011$ | $0.837 \pm 0.009$ | $0.828 \pm 0.010$ | $0.848 \pm 0.013$ |
|  | 9 s | $0.461 \pm 0.001$ | $0.759 \pm 0.004$ | $0.796 \pm 0.004$ | $0.841 \pm 0.005$ | $0.833 \pm 0.003$ | $0.856 \pm 0.002$ |
| M-LSTM | 5 s | $0.487 \pm 0.010$ | $0.761 \pm 0.003$ | $0.816 \pm 0.010$ | $0.856 \pm 0.011$ | $0.845 \pm 0.009$ | $0.859 \pm 0.008$ |
|  | 7 s | $0.470 \pm 0.012$ | $0.754 \pm 0.010$ | $0.818 \pm 0.005$ | $0.858 \pm 0.007$ | $0.845 \pm 0.009$ | $0.853 \pm 0.015$ |
|  | 9 s | $0.481 \pm 0.009$ | $0.767 \pm 0.011$ | $0.817 \pm 0.011$ | $0.856 \pm 0.008$ | $0.844 \pm 0.007$ | $0.858 \pm 0.008$ |
| S-LSTM | 5 s | $0.525 \pm 0.014$ | $0.787 \pm 0.006$ | $0.830 \pm 0.005$ | $0.872 \pm 0.003$ | $0.866 \pm 0.003$ | $0.892 \pm 0.004$ |
|  | 7 s | $\mathbf{0 . 5 2 6} \pm 0.004$ | $\mathbf{0 . 7 8 7} \pm 0.002$ | $\mathbf{0 . 8 3 9} \pm 0.014$ | $\mathbf{0 . 8 7 7} \pm 0.010$ | $\mathbf{0 . 8 7 2} \pm 0.006$ | $\mathbf{0 . 8 9 7} \pm 0.002$ |
|  | 9 s | $0.500 \pm 0.008$ | $0.777 \pm 0.015$ | $0.825 \pm 0.007$ | $0.866 \pm 0.008$ | $0.861 \pm 0.007$ | $0.888 \pm 0.006$ |

---

## Page 3

**Page Dimensions:** 1700x2200 (DPI: 200)

# 1.8 Analysis of Results 

The experimental results indicated that the S-LSTM variant consistently outperformed the other configurations, achieving the highest Balanced Accuracy, Cohen's Kappa, and F1 scores across all tested sample lengths. The Linear Transformer configuration showed the lowest performance in most metrics, suggesting that integrating xLSTM components substantially enhances the model's capability to capture temporal dependencies in biosignal data. Compared to the original BIOT implementation, our S-LSTM variant improved performance metrics by a notable margin, indicating the effectiveness of the exponential gating and memory mixing introduced by the xLSTM architecture. Conversely, the M-LSTM and M+S-LSTM configurations showed mixed results, generally performing better than the Linear Transformer but consistently below the S-LSTM configuration.

## References

[1] Yang, C., Westover, M. B., \& Sun, J. (2023). BIOT: Biosignal Transformer for Cross-data Learning in the Wild. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=c2LZyTyddi
[2] Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., \& Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. In Thirty-eighth Conference on Neural Information Processing Systems. https://arxiv.org/ abs/2405.04517

---

