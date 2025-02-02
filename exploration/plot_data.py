# %%
import mne
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
print(os.getcwd())
# %%
# Load an EEG file (replace with your file path)
file_path_abnormal = "datasets/TUAB/edf/train/abnormal/01_tcp_ar/aaaaaaaq_s004_t000.edf"
raw = mne.io.read_raw_edf(file_path_abnormal, preload=True)
spectrum = raw.compute_psd()
# Plot the raw data
raw.plot()

# %%

raw.info
# %%
spectrum.plot(average=True, picks="data", amplitude=False)

# %%
file_path_normal = "datasets/TUAB/edf/train/normal/01_tcp_ar/aaaaaaav_s004_t000.edf"
raw = mne.io.read_raw_edf(file_path_normal, preload=True)
spectrum = raw.compute_psd()
# Plot the raw data
raw.plot()

# %%
spectrum.plot(average=True, picks="data", amplitude=False)
# %%
file_path_normal = "datasets/TUEV/edf/train/aaaaaaar/aaaaaaar_00000001.edf"
raw = mne.io.read_raw_edf(file_path_normal, preload=True)
spectrum = raw.compute_psd()
# Plot the raw data
spectrum = raw.compute_psd()
raw.plot()

# %%
spectrum.plot(average=True, picks="data", amplitude=False)
# %%
raw.filter(l_freq=1.0, h_freq=40.0)

# %%
eventData = np.genfromtxt("datasets/TUEV/edf/train/aaaaaaar/aaaaaaar_00000001.rec", delimiter=",")
eventData
# %%