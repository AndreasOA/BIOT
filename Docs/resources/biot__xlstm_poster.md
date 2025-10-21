# Extracted PDF Content

Total pages: 1

---

## Page 0

**Page Dimensions:** 3979x2815 (DPI: 85)

# **Evaluating xLSTM on BIOT**

**Author:** Andreas Oberdammer **Supervisor:** Philipp Seidl **Institute for Machine Learning** **Johannes Kepler University** Linz, June 2025

## **INTRODUCTION**

**Electroencephalography (EEG)** classification is vital for **epilepsy diagnostics**. We evaluate how replacing the Biosignal Transformer's (BIOT) linear transformer block with Extended-LSTM variants — **mLSTM**, **sLSTM**, and **m+sLSTM** — affects performance on EEG windows of different lengths, highlighting the impact of sequence model choice.

## **HYPOTHESIS**

**Substituting** the linear transformer in BIOT with **xLSTM variants** will enhance the model's capability to capture **temporal dependencies** in EEG signals, **improving classification performance**.

## **DATASET**

We utilized the events dataset (**TUEV**), a subset of the **Temple University EEG corpus**. This dataset is **highly unbalanced** as only **23%** of training labels are used to **classify seizures**. The other labels are eye movement or other unrelated events (artifact and background).

|  EEG-Type | Sleep and resting  |
| --- | --- |
|  Nr. of recordings | 11914  |
|  Sampling Rate | 256 Hz  |
|  Channels | 22  |
|  Task type | Multi-class (6 event types)  |

- **Spike and Slow Wave** (SPSW)
- **Generalized Periodic Epileptiform Discharges** (GPED)
- **Periodic Lateralized Epileptiform Discharges** (PLED)
- Eye Movement (EYEM)
- Artifact (ARTF)
- Background (BCKG)

![img-0.jpeg](img-0.jpeg)

## **EXPERIMENT**

![img-1.jpeg](img-1.jpeg)

![img-2.jpeg](img-2.jpeg)

![img-3.jpeg](img-3.jpeg)

![img-4.jpeg](img-4.jpeg)

## **RESULTS**

During training we monitored **loss, balanced accuracy, weighted F1** and **cohen kappa**. Due to instable validation loss, we decided to select the best model based on the balanced accuracy on the validation set. We evaluated each model on multiple **sample lengths**, but for **simplicity** we decided to only display the **best result for each model.**

|  BIOT with | BACC | F1 | Cohen κ | AUCPR-Macro  |
| --- | --- | --- | --- | --- |
|  Linear Transformer (5s) | 0.5421 +/- 0.039 | 0.7331 +/- 0.007 | 0.4886 +/- 0.019 | 0.4752 +/- 0.011  |
|  sLSTM (7s) | 0.5540 +/- 0.025 | 0.7637 +/- 0.007 | 0.5499 +/- 0.017 | 0.5324 +/- 0.012  |
|  mLSTM (7s) | 0.4946 +/- 0.039 | 0.7015 +/- 0.018 | 0.4285 +/- 0.030 | 0.4649 +/- 0.017  |
|  sLSTM + mLSTM (7s) | 0.5058 +/- 0.035 | 0.7052 +/- 0.007 | 0.4395 +/- 0.019 | 0.4690 +/- 0.014  |

Values in parentheses indicate the sample length (5s / 7s) that yielded the best performance for each model after tuning.

## **References:**

[1] Tong, C., et al. (2023). BIOT: Biosignal Transformer for Cross-data Learning in the Wild. In *Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[2] Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory. *arXiv*. https://arxiv.org/abs/2495.04993

[3] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[4] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[5] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[6] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[7] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[8] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[9] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[10] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[11] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[12] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[13] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[14] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[15] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[16] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[17] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[18] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[19] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[20] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[21] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[22] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[23] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[24] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[25] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[26] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[27] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[28] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[29] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[30] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[31] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[32] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[33] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[34] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[35] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[36] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[37] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[38] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[39] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[40] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[41] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[42] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[43] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[44] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[45] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[46] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[47] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[48] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[49] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[50] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[51] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[52] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[53] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[54] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[55] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[56] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[57] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[58] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[59] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[60] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[61] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[62] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[63] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[64] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[65] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[66] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[67] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[68] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[69] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[70] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[71] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[72] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[73] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[74] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[75] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[76] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[77] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[78] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[79] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[80] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[81] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[82] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[83] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[84] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[85] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[86] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[87] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[88] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[89] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[90] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[91] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[92] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[93] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[94] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[95] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[96] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[97] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[98] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[99] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[100] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[101] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[102] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[103] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[104] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[105] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[106] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[107] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[108] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[109] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1101] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1112] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1122] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[113] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[114] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[115] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[116] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[117] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[118] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[119] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[120] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1210] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[12210] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[12210] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[12310] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[12410] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1250] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1250] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1250] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1260] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1270] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1280] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1290] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1300] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1310] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1330] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1330] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1310] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, https://openreview.net/forum?dno3LZyTyddi

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the Twenty-secds

[1320] Song, Y., et al. (2023). *BIOT: A Neural Transformer for Cross-Data Learning in the Wild*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 2023*. In *Proceedings of the 

**Images on this page:** 5
- Image ID: img-0.jpeg (Position: 771,2263 to 1309,2781)
- Image ID: img-1.jpeg (Position: 1273,748 to 1925,1801)
- Image ID: img-2.jpeg (Position: 1973,748 to 2351,1801)
- Image ID: img-3.jpeg (Position: 2383,748 to 2944,1801)
- Image ID: img-4.jpeg (Position: 3059,748 to 3931,1801)

---

