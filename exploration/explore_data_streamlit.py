import streamlit as st
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_eeg_with_events(signals, times, events, selected_channels, start_time=None, end_time=None):
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Define colors for each label
    label_colors = {
        1: 'red',    # spsw
        2: 'blue',   # gped
        3: 'green',  # pled
        4: 'yellow', # eyem
        5: 'purple', # artf
        6: 'gray'    # bckg
    }
    
    # Define label names for the legend
    label_names = {
        1: 'spsw',
        2: 'gped',
        3: 'pled',
        4: 'eyem',
        5: 'artf',
        6: 'bckg'
    }
    
    num_channels = len(selected_channels)
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3 * num_channels), sharex=True)
    
    for idx, ax in enumerate(axes):
        channel_index = selected_channels[idx]
        
        # Plot each channel's signal using Seaborn
        sns.lineplot(x=times, y=signals[channel_index], ax=ax, color='black', label=f'Channel {channel_index + 1}')
        
        # Overlay the events with colored backgrounds
        for event in events:
            channel, start_time_event, stop_time_event, label = event
            if int(channel) == channel_index + 1:  # Check if the event is for the current channel
                color = label_colors.get(int(label), 'black')  # Default to black if label not found
                ax.axvspan(start_time_event, stop_time_event, color=color, alpha=0.3)
        
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Channel {channel_index + 1}')
    
    # Create custom legend for event labels
    handles = [plt.Line2D([0], [0], color=color, lw=4, label=label_names[label]) 
               for label, color in label_colors.items()]
    fig.legend(handles=handles, loc='upper right', title='Event Labels')
    
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make space for sliders

    # Set x-axis limits if start_time and end_time are provided
    if start_time is not None and end_time is not None:
        for ax in axes:
            ax.set_xlim(start_time, end_time)

    st.pyplot(fig)


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName)
    signals, times = Rawdata[:]
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]

def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  # 0
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  # 1
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  # 2
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  # 4
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  # 5
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  # 6
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 7
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  # 14
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  # 15
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  # 16
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 17
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  # 18
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  # 19
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21
    return new_signals

def main():
    st.set_page_config(layout="wide")
    st.title("EDF File Viewer")

    st.write("""
    ### Label Information:
    <div style="color: red;"><strong>1. spsw (spike and slow wave): </strong></div> This pattern is often associated with epileptic activity. Spikes and slow waves are common in EEGs of individuals with epilepsy and can indicate seizure activity or a predisposition to seizures.
    
    <div style="color: blue;"><strong>2. gped (generalized periodic epileptiform discharge): </strong></div> These discharges are indicative of generalized epileptic activity. They are often seen in conditions like generalized epilepsy and can be associated with ongoing or impending seizures.
    
    <div style="color: green;"><strong>3. pled (periodic lateralized epileptiform discharge): </strong></div> PLEDs are typically associated with focal seizures and can indicate localized brain dysfunction. They are often seen in acute neurological conditions and can be a precursor to seizures.
    
    <div style="color: yellow;"><strong>4. eyem (eye movement): </strong></div> Eye movements are not directly related to seizures but can appear as artifacts in EEG recordings. They are important to identify and differentiate from epileptic activity.
    
    <div style="color: purple;"><strong>5. artf (artifact): </strong></div> Artifacts are non-neurological signals that can contaminate EEG recordings. They are not related to seizures but need to be distinguished from true epileptic activity to avoid misinterpretation.
    
    <div style="color: gray;"><strong>6. bckg (background): </strong></div> This label is used for EEG segments that do not contain any of the specific epileptic or non-epileptic events listed above. It serves as a catch-all category for normal or non-specific EEG activity.
    """, unsafe_allow_html=True)

    # Sidebar for form options
    with st.sidebar:
        # Get list of folders in the train directory
        train_folders = sorted(os.listdir("datasets/TUEV/edf/train"))
        
        # Create a select box for folder selection
        selected_folder = st.selectbox("Select a folder", train_folders)
        
        # Full path to the selected EDF file
        folder_path = os.path.join("datasets/TUEV/edf/train", selected_folder)
        edf_files = [f for f in os.listdir(folder_path) if f.endswith('.edf')]
        
        # Initialize session state for file loading
        if 'file_loaded' not in st.session_state:
            st.session_state.file_loaded = False
            st.session_state.signals = None
            st.session_state.times = None
            st.session_state.eventData = None
            st.session_state.Rawdata = None

        # Add a submit button to load the file
        if st.button("Load EDF File") or not st.session_state.file_loaded:
            # Read and process the EDF file
            signals, times, eventData, Rawdata = readEDF(folder_path + "/" + edf_files[0])
            new_signals = convert_signals(signals, Rawdata)
            
            # Store data in session state
            st.session_state.file_loaded = True
            st.session_state.signals = new_signals
            st.session_state.times = times
            st.session_state.eventData = eventData
            st.session_state.Rawdata = Rawdata
            
            # Display some information about the file
            st.write(f"Loaded file: {edf_files[0]}")
            st.write(f"Number of channels: {new_signals.shape[0]}")
            st.write(f"Duration: {times[-1]} seconds")
        
        if st.session_state.file_loaded:
            # Multi-select for channels
            channel_options = [f"Channel {i+1}" for i in range(st.session_state.signals.shape[0])]
            selected_channels = st.multiselect("Select channels to plot", channel_options, default=channel_options)
            
            # Convert selected channel names to indices
            selected_indices = [channel_options.index(ch) for ch in selected_channels]
            
            start_time, end_time = st.slider(
                "Select time range",
                min_value=0.0,
                max_value=float(st.session_state.times[-1]),
                value=(0.0, float(st.session_state.times[-1])),
                step=0.1
            )
            
            # Button to plot the data
            plot_button = st.button("Plot EEG with Events")

    # Plot on the main page
    if st.session_state.file_loaded and plot_button:
        plot_eeg_with_events(st.session_state.signals, st.session_state.times, st.session_state.eventData, selected_indices, start_time, end_time)

if __name__ == "__main__":
    main()