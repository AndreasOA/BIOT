The Temple University Hospital EEG Corpus:
Electrode Location and Channel Labels
April 6, 2022
Prepared By:
Sean Ferrell, Vineetha Mathew, Matthew Refford, Vincent Tchiong,
Tameem Ahsan, Iyad Obeid and Joseph Picone
The Neural Engineering Data Consortium
College of Engineering, Temple University 
1947 North 12th Street
Philadelphia, Pennsylvania 19122
Tel: 215-204-4841; Fax: 215-204-5960
Email: {sean.ferrell, vineetha.mathew, matthew.refford, vincent.tchiong,
tameem.ahsan, iobeid, picone}@temple.edu

 
ABSTRACT
The goal of this report is to describe to users of the TUH EEG Corpus four important concepts that must be understood to correctly retrieve EEG signals from a data file (e.g., an EDF file). The four key concepts described in this document are: (1) physical placement: the location of the electrodes on the scalp, (2) unipolar montage: the differential recording process used to reduce noise, (3) channel labels: the system used to describe the channels, or digital signals, represented in a computer file and (4) bipolar montages: the differential mapping used to accentuate clinically-relevant events in the signal. This report is not intended to be a primer on the electrophysiology of an EEG, which is a subject unto itself, or a tutorial on how neurologists interpret EEGs. This report simply explains how the signal data in an EEG file must be accessed to accurately support clinical applications (e.g., manual interpretation or annotation of an EEG) and research applications (e.g., automatic interpretation using machine learning).
1.	INTRODUCTION
An electroencephalogram (EEG) (Tong & Thakor, 2009) is a vital tool for monitoring the brain’s electrical activity and diagnosing various neural diseases. The Temple University Hospital EEG (TUEG) Corpus (Obeid & Picone, 2016) was developed to support state of the art research into automatic interpretation of EEGs using machine learning (Golmohammadi et al., 2019). The corpus consists of EEG signal data and paired EEG reports written by the attending neurologist for each patient session. The signal data is stored in an open-source European Data Format (EDF) file, (Kemp, 2013) while the reports are stored as flat text files. The header of each EDF file contains fundamental metadata information about every patient session that is evenly distributed over 24 fields that display patient information and signal condition. There are several other valuable subsets and annotations of the data (Veloso et al., 2017) that are available from our project web site (Choi et al., 2017).
This corpus consists entirely of clinical data collected from 2002 to the present at Temple University Hospital (TUH). The EEG signal data was collected using various generations of EEG equipment. Most of the data from 2002 – 2019 was collected using Natus Medical Incorporated’s (NMI) NicoletOne recording equipment (Natus, 2019). Unfortunately, this system stores the data in a proprietary format developed by Natus. The signal data was exported from the source EEG files to an open source publicly accessible format using NMI’s proprietary NicVue software tool (NicVue, 2019). The signal data that is stored represents a pruned EEG in which sections of the EEG signal marked as uninformative by the attending technician were removed. One or more EDF files are generated from a signal source file based on these pruning instructions provided by the attending technician. A direct result of this is that the Natus tool only outputs pruned data, so we had no way of releasing the entire original recording. This is still an area under research as we continue to evaluate open source tools that claim to be able to reverse engineer the Natus proprietary format.
A point we have continually emphasized throughout our long history with EEG technology development is that clinical data is messy. The data in TUEG was collected from a variety of locations in the hospital including the intensive care unit (ICU), the epilepsy monitoring unit (EMU), the emergency room (ER) and outpatient services (the 5th floor of Boyer Pavilion). Because these sessions cover the full range of EEGs conducted at TUH, there are a broad range of channel configurations and channel labels used to describe the EEG data. In fact, there are over 40 unique channel configurations contained within the entire corpus. The EDF format uses an ASCII representation for the header contained within the file while the signal data is stored as a multichannel signal in which samples are encoded as 16-bit integers. Signal channels within this file are described by labels that can be used to infer the original physical location of the electrode and the meaning of the channel.
There is no guarantee that the channels that comprise an EEG signal appear in the file in the same order. A common mistake made by many rudimentary software packages is that they read the data into a matrix and assume the channels are always in the same order. A key portion of an EDF file that contains the labels is shown in Figure 1. As can be seen, each channel is labeled, and each channel must be accessed via its label and/or position in this list rather than its absolute position in the file. For example, there is no constraint that the channel labeled “EEG FP1-REF” is always the first channel. This channel can appear in many positions across TUEG.
Therefore, the primary goal of this report is to document the labels that appear in this corpus and to explain how these can be mapped back to the physical locations of the sensors. We have developed a visualization tool (Capp et al., 2017) that simplifies visualizing and manipulating these channels by their labels. We provide a program, nedc_pystream, that is easy to use and demonstrates how to correctly access the data. This Python code is available from our project web site (Choi et al., 2017).
Most commercial packages offer similar capabilities, since clinicians need to be able to manipulate channels symbolically. Interpreting labels can only be done through auxiliary documentation, such as that provided in this report. Channel labels, though often similar across institutions, are not guaranteed to be common across institutions (e.g., institution X might not use the label “EEG FP1-REF”). Only through documentation such as that provided in this report, can one reverse map the data in an EDF file.
When an EEG is administered, a technician wires up a patient with a specific electrode configuration. This includes deciding on the number of channels to be collected, the locations of the electrodes on the scalp, and the reference points used for the electrical signals. These “raw” signals, which are digitized versions of the electrical potential measured between an electrode and a reference point, are stored in an EEG file. We discuss the issues of electrode placement, which we refer to in this document as the physical configuration, in Section 2. We discuss the process of differential voltage recording, which is referred to as a unipolar montage, in Section 3.
Each channel that is recorded is identified by a label (e.g., “EEG FP1-REF”) as shown in Figure 1. These labels, unfortunately, are specific to an institution, neurologist or technician. We discuss the interpretation of channel labels in Section 4. In Appendix B we list every channel label that appears at least once in the corpus, and the number of files in which it appears. The specific electrode location that corresponds to each of these labels is not known since most of this data was collected long before we engaged Temple Hospital, and the associated documentation has been lost over time. Nevertheless, we have attempted to provide some useful information about the origin of these labels.
In Appendix C we provide the channel labels appearing in the Duke University (DUSZ) Corpus (Swisher et al., 2015). This is a corpus similar to TUEG that used different electronic equipment to collect the EEG signal. It is valuable for testing the ability of machine learning models to assess cross-channel robustness.
It is also important to understand that when neurologists view an EEG, they impose a montage on the data. This is most often simply a list of channel pairs to be differenced and is commonly referred to as a bipolar montage. This type of montage is described in detail in Section 5. Note that this differencing is in addition to the differencing done to record the raw signal. A bipolar montage is imposed when the data is viewed or processed and is not actually stored in the data file. Neurologists will often view the same EEG using several different montages depending on what specific events they are looking to enhance. Software that loads an EEG and displays the results is responsible for implementing the montage view. This software must provide some way of directing which channels are to be differenced. As mentioned previously, we provide software that demonstrates how to do this in Python. Most commercial packages offer similar capabilities, since these bipolar montages are used heavily in clinical work. Although there are many popular montages used in practice (e.g., TCP), some expert clinicians prefer their own definitions. Imposing a bipolar montage is critical for machine learning researchers since they will often use the output of a bipolar montage as a starting point for their algorithm research.
2.	PHYSICAL CONFIGURATION OF THE ELECTRODES
The most widely used arrangement of electrodes in electroencephalographic recordings is the International 10/20 system. In this system, 21 electrodes on the scalp are evenly distributed as seen in Figure 2 (López, 2017), with the distance between electrodes being either 10% or 20% of the total distance from nasion (front) to inion (back). The 10/20 system utilizes four anatomical landmarks for positioning: the point between the forehead and nose (nasion), the lowest point of the back of the skull (inion), and the preauricular areas anterior to the ears.
The 10/10 and 10/5 electrode configurations are extensions of the original standard 10/20 system. The 10 10 system derives its electrode placements from landmarks on the skull, such as the nasion (Nz), and the inion (Iz), and the left and right pre-auricular points (LPA and RPA). The 10/10 system differs from the 10/20 system in that the number of electrodes used increases from 21 to 74. In order to account for an increased number of electrodes, the number of locations have to be extended in all directions to keep an even spread of data extracted from the patient’s scalp. However, the 10/10 and 10/5 system are not as widely used as the 10/20 system, especially for clinical applications. As a result, the bulk of our data uses the 10/20 system.
3.	UNIPOLAR MONTAGES USED FOR RECORDING
When recording an EEG signal and writing it to a file as a digital signal represented using 16 bits per sample, a differential voltage must be recorded. The need for this relates to the electrophysiology of an EEG, which is outside the scope of this document. Suffice it to say that for these types of low-voltage signals (typically in the microvolt range) to be useful, differential voltages must be used because the differencing process reduces noise. A unipolar montage refers to the difference between the electrical potential recorded at an electrode, which we refer to as the raw signal, and a reference node (e.g., an electrode connected to the left ear). This differential signal is what is recorded as a digital signal in a data file. All channels are collected as differential voltages, so a configuration of reference points is implied by the data written in a file.
Two general unipolar montages, which are shown in Figure 3, are used within TUEG: (1) Average Reference (AR) and (2) Linked Ears Reference (LE). The AR montage uses the average of a certain number of electrodes as the reference. The LE montage uses a lead adapter to link the left and right ears, providing a more stable reference point (Lopez et al, 2016). The LE montage is believed to reduce artifacts (Subramaniyam, 2019).
The LE and AR unipolar montages are divided into four classifications in TUEG: 01_tcp_ar, 02_tcp_le, 03_tcp_ar_a, and 04_tcp_le_a. Each of these montages is based on the types of channels included. The montage labeled 01_tcp_ar uses the AR referencing method for the electrodes. The montage 02_tcp_le classification uses the LE montage format. The unipolar montages labeled as 03_tcp_ar_a and 04_tcp_le_a use AR and LE formats respectively but are collected with only 20 channels. For these last two subgroups, the auricular channels are excluded (electrodes A1 and A2).
4.	CHANNEL LABELS
Each channel in an EDF file is labeled using a non-standard set of labels. A typical set of these labels is shown in Figure 1. As mentioned previously, a complete set of labels is listed in Appendix B. These labels unfortunately are non-standard. However, the names are often adequately descriptive so that a user can infer the nature and location of the digitized signal. It is important to understand that to access data in a correct and consistent manner, you must pay attention to the channel label. You cannot assume that the first channel stored in an EDF file always represents the same electrode location.
Each electrode begins with a letter corresponding to the region where the signals are read from:
•	Fp: Prefrontal	•	P: Parietal/Parasagittal
•	F: Frontal	•	O: Occipital
•	T: Temporal	•	A: Pre-auricular
•	C: Central
Even numbers (2, 4, 6, 8) are used to denote electrodes in the right hemisphere and odd numbers (1, 3, 5, 7) refer to those on the left. Each adjacent electrode represents a distance of 10% or 20% of either the total nasion-inion or right-left distance, hence the 10-20 system. “Z” refers to electrodes located on the midsagittal line. For example, Fp1-F3 refers to the signal in the segment between the left prefrontal electrode closest to the nasion and the left frontal electrode closest to the midsagittal line. The signal would represent the brain activity in the imaginary line between these two electrodes.
A complete listing of the channels associated with each of the four unipolar montages is given in Appendix A. Software is available on our project web site (Choi et al., 2017) that demonstrates one way to decode channels properly in Python. Our interface supports simple pattern matching so that channel labels can be easily identified. The software also supports a straightforward way of specifying a montage. Several of our software tools use this format, including our annotation tool (Capp et al., 2017).
Many EEG records in TUEG have at least 19 electrodes, corresponding to the aforementioned 10/20 system. Table 1 lists these electrodes with their corresponding labels. Table 1 also provides a brief description of the location of each electrode. In the AR and LE montages there are 22 signal channels that are derived from the 19 electrodes; this is due to the T3 and T4 electrodes being used twice, both longitudinally and transversely. Along the midsagittal line in the 10/20 system there are five central “Z” channels (Fpz, Fz, Cz, Pz, Oz). However, the bipolar montages applied to the data in TUEG only reference Cz, located at the apex of the scalp.
Though there are many labels listed in Appendix B, most researchers will probably not use most of these. The labels in Table 1 are most of the useful labels for machine learning research. Our software interface supports partial name matching, which makes dealing with these labels much easier.
5.	BIPOLAR MONTAGES USED FOR VIEWING
As mentioned previously, differential voltages are used to reduce noise and enhance events of interest, such as spikes. The electrical signal in the area between adjacent electrodes cancels out noise and artifacts that are due to a common reference point. This often results in a clearer and more easily interpretable signal. However, this method also makes certain electrode combinations more vulnerable to specific artifacts. 
When neurologists view the data, they typically impose a bipolar montage to remove signal noise and improve spatial information interpretation of the EEG signal (Shah et al., 2017). We similarly use one of the most popular bipolar Temporal Central Parasagittal (TCP) montages for EEG interpretation and algorithm development. A TCP montage is also known as the double-banana montage. It is shown in Figure 4. This montage uses signals that correspond to the difference between two adjacent electrodes (e.g. FP1¬ F7, T3-C3), in the nasion-inion/longitudinal direction, or transverse across the scalp (left-to-right).
There are a wide range of other montages in use at the TUH. However, the TCP bipolar montage is by far the most popular among neurologists.
6.	FILENAMING CONVENTIONS
The corpus is stored using a descriptive filename that makes it easy to locate subsets of data using standard Unix commands. A typical directory will look something like this:
nedc_000_[1]: p
/data/isip/data/tuh_eeg/v1.1.0/edf/01_tcp_ar/130/00013001/s001_2015_08_12
nedc_000_[1]: d
total 18571
drwxrwxr-x 2 picone isip        5 Nov 19  2020 ./
drwxrwxr-x 3 picone isip        3 Nov 19  2020 ../
-rw-rw-r-- 1 picone isip  1856984 Nov 19  2020 00013001_s001_t001.edf
-rw-rw-r-- 1 picone isip 16955452 Nov 19  2020 00013001_s001_t002.edf
-rw-rw-r-- 1 picone isip     1586 Nov 19  2020 00013001_s001.txt
This directory contains two EDF files that contain the pruned signal data from the original recording session and a text file (“*.txt”) that contains a plain text version of the EEG report. Corpora that have been annotated will also contain “*.csv” and “*.xml” files that contain annotation information. See Ochal et al. (2020) for more information on this.
The pathname and filename used to represent the data are shown in Table 1. Each file has a unique name that includes the medical record number (MRN), session number and token number. As explained previously, EEG recordings are pruned or split into multiple files. The token number is used to represent these multiple files. Though most of the time they are in sequence (e.g., s002 occurs later in time than s001), the exact ordering of these files depends somewhat on how the technician labeled the data. This is out of our control unfortunately. In our research, we treat each edf file as an independent event.
7.	SUMMARY
The goal of this report is to document how EEG signal data was collected and stored in EDF files for TUEG. We have described how channel ordering, channel labels, and the physical location of a channel can be cross referenced through the use of information available in this document and in the header of the EDF file. We have documented the four main montages used in TUEZ and presented an exhaustive list of channel labels. In a companion document we discuss the process of annotating EEG signals and storing these annotations in text files (Ochal et al., 2020).
To accurately process EEG data in TUEG, you must pay attention to channel ordering and labels. We have also developed software that demonstrates how this can be done in Python. This software is available from our project web site.
To access these resources, we encourage you to register on our project web site:
https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
Enrollment is quick and completely automated. You will receive a username and password that will let you access all of our online resources.
Questions about this document or our resources should be directed to help@nedcdata.org.
ACKNOWLEDGEMENTS
Research reported in this publication was most recently supported by the National Science Foundation Partnership for Innovation award number IIP-1827565 and the Pennsylvania Commonwealth Universal Research Enhancement Program (PA CURE). 
Several grants over the years have supported this database development project. Significant contributors include National Human Genome Research Institute of the National Institutes of Health award number U01HG008468, DARPA Microsystems Technology Office award number D13AP00065, National Science Foundation Division of Computer and Network Systems award number CNS-1305190, the Temple University Office of the Vice-Provost for Research and the Temple University College of Engineering.
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the official views of any of these organizations.
REFERENCES
Capp, N., Krome, E., Obeid, I., & Picone, J. (2017). Facilitating the annotation of seizure events through an extensible visualization tool. In I. Obeid & J. Picone (Eds.), IEEE Signal Processing in Medicine and Biology Symposium (p. 1). Philadelphia, Pennsylvania, USA: IEEE. https://doi.org/10.1109/SPMB.2017.8257043.
Choi, S. I., Lopez, S., Obeid, I., Jacobson, M., & Picone, J. (2017). The Temple University Hospital EEG Corpus. http://www.isip.piconepress.com/projects/tuh_eeg.
Golmohammadi, M., Harati Nejad Torbati, A., Lopez, S., Obeid, I., & Picone, J. (2019). Automatic Analysis of EEGs Using Big Data and Hybrid Deep Learning Architectures. Frontiers in Human Neuroscience, 13(76), 1–30. https://doi.org/10.3389/fnhum.2019.00076.
Kemp, R. (2013). European Data Format. Retrieved from http://www.edfplus.info.
Lopez, S., Golmohammadi, M., Obeid, I., & Picone, J. (2016). An analysis of two common reference points for EEGs. In I. Obeid & J. Picone (Eds.), Proceedings of the IEEE Signal Processing in Medicine and Biology Symposium (pp. 1–4). Philadelphia, Pennsylvania, USA. https://doi.org/10.1109/SPMB.2016.7846854.
Lopez, S., Suarez, G., Jungreis, D., Obeid, I., & Picone, J. (2015). Automated Identification of Abnormal EEGs. In I. Obeid & J. Picone (Eds.), IEEE Signal Processing in Medicine and Biology Symposium (pp. 1–4). Philadelphia, Pennsylvania, USA: IEEE. https://doi.org/10.1109/SPMB.2015.7405423.
Natus Medical. (2018). Nicolet® NicVue Connectivity Solution. Retrieved from          https://neuro.natus.com/products-services/nicolet-nicvue-connectivity-solution.
Natus Medical. (n.d.). NicoletOneTM LTM System. Retrieved from https://neuro.natus.com/products-services/nicoletone-ltm-system.
Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data Corpus. Frontiers in Neuroscience, Section Neural Technology, 10, 00196. http://dx.doi.org/10.3389/fnins.2016.00196.
Ochal, D., Rahman, S., Ferrell, S., Elseify, T., Obeid, I., & Picone, J. (2020). The Temple University Hospital EEG Corpus: Annotation Guidelines. https://www.isip.piconepress.com/publications/reports/2020/tuh_eeg/annotations/.
Shah, V., Golmohammadi, M., Ziyabari, S., von Weltin, E., Obeid, I., & Picone, J. (2017). Optimizing Channel Selection for Seizure Detection. In I. Obeid & J. Picone (Eds.), Proceedings of the IEEE Signal Processing in Medicine and Biology Symposium (pp. 1–5). Philadelphia, Pennsylvania, USA: IEEE. https://doi.org/10.1109/SPMB.2017.8257019.
Subramaniyam, N. P. (2019). Effect of EEG Reference Choice on Outcomes. https://sapienlabs.co/effect-of-eeg-reference-choice-on-outcomes/. 
Swisher, C. B., Shah, D., Sinha, S. R., & Husain, A. M. (2015). Baseline EEG Pattern on Continuous ICU EEG Monitoring and Incidence of Seizures. Journal of Clinical Neurophysiology, 32(2), 147–151. https://doi.org/10.1097/WNP.0000000000000157.
Tong, S., & Thakor, N. (2009). Quantitative EEG analysis methods and clinical applications. (S. Ton & N.  Thakor, Eds.). New York, New York, USA: Artech House
Veloso, L., McHugh, J. R., von Weltin, E., Obeid, I., & Picone, J. (2017). Big Data Resources for EEGs: Enabling Deep Learning Research. In I. Obeid & J. Picone (Eds.), Proceedings of the IEEE Signal Processing in Medicine and Biology Symposium (p. 1). Philadelphia, Pennsylvania, USA: IEEE. https://doi.org/10.1109/SPMB.2017.8257044.
Appendix A.	Unipolar Montages Used in TUEG
[1]	TCP_AR
[2]	TCP_LE
[3]	TCP_AR_A
[4]	TCP_LE_A
Appendix B.	Channel Labels Appearing in TUEG (v1.1.0)
Index	Label	Freq	Description
1	BURSTS	37,546 	unknown
2	DC1-DC	8,047	DC voltage equipment
3	DC2-DC	8,047	DC voltage equipment
4	DC3-DC	8,047	DC voltage equipment
5	DC4-DC	8,047	DC voltage equipment
6	DC5-DC	8,047	DC voltage equipment
7	DC6-DC	8,047	DC voltage equipment
8	DC7-DC	8,047	DC voltage equipment
9	DC8-DC	8,047	DC voltage equipment
10	ECG EKG-REF	70	a single ECG electrode placed on the chest to monitor cardiac activity
11	EDF ANNOTATIONS	475	annotations created by EEG technician
12	EEG 100-REF	323	used in custom and high-resolution systems (no signal data)
13	EEG 101-REF	323	used in custom and high-resolution systems (no signal data)
14	EEG 102-REF	323	used in custom and high-resolution systems (no signal data)
15	EEG 103-REF	323	used in custom and high-resolution systems (no signal data)
16	EEG 104-REF	323	used in custom and high-resolution systems (no signal data)
17	EEG 105-REF	323	used in custom and high-resolution systems (no signal data)
18	EEG 106-REF	323	used in custom and high-resolution systems (no signal data)
19	EEG 107-REF	323	used in custom and high-resolution systems (no signal data)
20	EEG 108-REF	323	used in custom and high-resolution systems (no signal data)
21	EEG 109-REF	323	used in custom and high-resolution systems (no signal data)
22	EEG 110-REF	323	used in custom and high-resolution systems (no signal data)
23	EEG 111-REF	323	used in custom and high-resolution systems (no signal data)
24	EEG 112-REF	323	used in custom and high-resolution systems (no signal data)
25	EEG 113-REF	323	used in custom and high-resolution systems (no signal data)
26	EEG 114-REF	323	used in custom and high-resolution systems (no signal data)
27	EEG 115-REF	323	used in custom and high-resolution systems (no signal data)
28	EEG 116-REF	323	used in custom and high-resolution systems (no signal data)
29	EEG 117-REF	323	used in custom and high-resolution systems (no signal data)
30	EEG 118-REF	323	used in custom and high-resolution systems (no signal data)
31	EEG 119-REF	323	used in custom and high-resolution systems (no signal data)
32	EEG 120-REF	323	used in custom and high-resolution systems (no signal data)
33	EEG 121-REF	323	used in custom and high-resolution systems (no signal data)
34	EEG 122-REF	323	used in custom and high-resolution systems (no signal data)
35	EEG 123-REF	323	used in custom and high-resolution systems (no signal data)
36	EEG 124-REF	323	used in custom and high-resolution systems (no signal data)
37	EEG 125-REF	323	used in custom and high-resolution systems (no signal data)
38	EEG 126-REF	323	used in custom and high-resolution systems (no signal data)
39	EEG 127-REF	323	used in custom and high-resolution systems (no signal data)
40	EEG 128-REF	323	used in custom and high-resolution systems (no signal data)
41	EEG 1X10_LAT_01-	43	custom electrode placement
42	EEG 1X10_LAT_02-	43	custom electrode placement
43	EEG 1X10_LAT_03-	43	custom electrode placement
44	EEG 1X10_LAT_04-	43	custom electrode placement
45	EEG 1X10_LAT_05-	42	custom electrode placement
46	EEG 20-LE	16	custom electrode placement
47	EEG 20-REF	2,585	custom electrode placement
48	EEG 21-LE	16	custom electrode placement
49	EEG 21-REF	2,612	custom electrode placement
50	EEG 22-LE	16	custom electrode placement
51	EEG 22-REF	2,612	custom electrode placement
52	EEG 23-LE	596	custom electrode placement
53	EEG 23-REF	2,585	custom electrode placement
54	EEG 24-LE	596	custom electrode placement
55	EEG 24-REF	2,585	custom electrode placement
56	EEG 25-LE	16	custom electrode placement
57	EEG 25-REF	2,714	custom electrode placement
58	EEG 26-LE	8,897	custom electrode placement
59	EEG 26-REF	8,135	custom electrode placement
60	EEG 27-LE	8,897	custom electrode placement
61	EEG 27-REF	8,018	custom electrode placement
62	EEG 28-LE	11,197	custom electrode placement
63	EEG 28-REF	8,022	custom electrode placement
64	EEG 29-LE	11,197	custom electrode placement
65	EEG 29-REF	11,325	custom electrode placement
66	EEG 30-LE	12,818	custom electrode placement
67	EEG 30-REF	11,328	custom electrode placement
68	EEG 31-LE	8,317	custom electrode placement
69	EEG 31-REF	17,837	custom electrode placement
70	EEG 32-LE	8,317	custom electrode placement
71	EEG 32-REF	17,837	custom electrode placement
72	EEG 33-REF	323	used in custom and high-resolution systems (no signal data)
73	EEG 34-REF	323	used in custom and high-resolution systems (no signal data)
74	EEG 35-REF	323	used in custom and high-resolution systems (no signal data)
75	EEG 36-REF	323	used in custom and high-resolution systems (no signal data)
76	EEG 37-REF	323	used in custom and high-resolution systems (no signal data)
77	EEG 38-REF	323	used in custom and high-resolution systems (no signal data)
78	EEG 39-REF	323	used in custom and high-resolution systems (no signal data)
79	EEG 40-REF	323	used in custom and high-resolution systems (no signal data)
80	EEG 41-REF	323	used in custom and high-resolution systems (no signal data)
81	EEG 42-REF	323	used in custom and high-resolution systems (no signal data)
82	EEG 43-REF	323	used in custom and high-resolution systems (no signal data)
83	EEG 44-REF	323	used in custom and high-resolution systems (no signal data)
84	EEG 45-REF	323	used in custom and high-resolution systems (no signal data)
85	EEG 46-REF	323	used in custom and high-resolution systems (no signal data)
86	EEG 47-REF	323	used in custom and high-resolution systems (no signal data)
87	EEG 48-REF	323	used in custom and high-resolution systems (no signal data)
88	EEG 49-REF	323	used in custom and high-resolution systems (no signal data)
89	EEG 50-REF	323	used in custom and high-resolution systems (no signal data)
90	EEG 51-REF	323	used in custom and high-resolution systems (no signal data)
91	EEG 52-REF	323	used in custom and high-resolution systems (no signal data)
92	EEG 53-REF	323	used in custom and high-resolution systems (no signal data)
93	EEG 54-REF	323	used in custom and high-resolution systems (no signal data)
94	EEG 55-REF	323	used in custom and high-resolution systems (no signal data)
95	EEG 56-REF	323	used in custom and high-resolution systems (no signal data)
96	EEG 57-REF	323	used in custom and high-resolution systems (no signal data)
97	EEG 58-REF	323	used in custom and high-resolution systems (no signal data)
98	EEG 59-REF	323	used in custom and high-resolution systems (no signal data)
99	EEG 60-REF	323	used in custom and high-resolution systems (no signal data)
100	EEG 61-REF	323	used in custom and high-resolution systems (no signal data)
   101	EEG 62-REF	323	used in custom and high-resolution systems (no signal data)
   102	EEG 63-REF	323	used in custom and high-resolution systems (no signal data)
   103	EEG 64-REF	323	used in custom and high-resolution systems (no signal data)
   104	EEG 65-REF	323	used in custom and high-resolution systems (no signal data)
   105	EEG 66-REF	323	used in custom and high-resolution systems (no signal data)
   106	EEG 67-REF	323	used in custom and high-resolution systems (no signal data)
   107	EEG 68-REF	323	used in custom and high-resolution systems (no signal data)
   108	EEG 69-REF	323	used in custom and high-resolution systems (no signal data)
   109	EEG 70-REF	323	used in custom and high-resolution systems (no signal data)
   110	EEG 71-REF	323	used in custom and high-resolution systems (no signal data)
   111	EEG 72-REF	323	used in custom and high-resolution systems (no signal data)
   112	EEG 73-REF	323	used in custom and high-resolution systems (no signal data)
   113	EEG 74-REF	323	used in custom and high-resolution systems (no signal data)
   114	EEG 75-REF	323	used in custom and high-resolution systems (no signal data)
   115	EEG 76-REF	323	used in custom and high-resolution systems (no signal data)
   116	EEG 77-REF	323	used in custom and high-resolution systems (no signal data)
   117	EEG 78-REF	323	used in custom and high-resolution systems (no signal data)
   118	EEG 79-REF	323	used in custom and high-resolution systems (no signal data)
   119	EEG 80-REF	323	used in custom and high-resolution systems (no signal data)
   120	EEG 81-REF	323	used in custom and high-resolution systems (no signal data)
   121	EEG 82-REF	323	used in custom and high-resolution systems (no signal data)
   122	EEG 83-REF	323	used in custom and high-resolution systems (no signal data)
   123	EEG 84-REF	323	used in custom and high-resolution systems (no signal data)
   124	EEG 85-REF	323	used in custom and high-resolution systems (no signal data)
   125	EEG 86-REF	323	used in custom and high-resolution systems (no signal data)
   126	EEG 87-REF	323	used in custom and high-resolution systems (no signal data)
   127	EEG 88-REF	323	used in custom and high-resolution systems (no signal data)
   128	EEG 89-REF	323	used in custom and high-resolution systems (no signal data)
   129	EEG 90-REF	323	used in custom and high-resolution systems (no signal data)
   130	EEG 91-REF	323	used in custom and high-resolution systems (no signal data)
   131	EEG 92-REF	323	used in custom and high-resolution systems (no signal data)
   132	EEG 93-REF	323	used in custom and high-resolution systems (no signal data)
   133	EEG 94-REF	323	used in custom and high-resolution systems (no signal data)
   134	EEG 95-REF	323	used in custom and high-resolution systems (no signal data)
   135	EEG 96-REF	323	used in custom and high-resolution systems (no signal data)
   136	EEG 97-REF	323	used in custom and high-resolution systems (no signal data)
   137	EEG 98-REF	323	used in custom and high-resolution systems (no signal data)
   138	EEG 99-REF	323	used in custom and high-resolution systems (no signal data)
   139	EEG A1-LE	12,802	left reference electrode in an LE montage located in the left preauricular area
   140	EEG A1-REF	34,633	left preauricular area
   141	EEG A2-LE	12,802	right reference electrode in an LE montage located in the right preauricular area
   142	EEG A2-REF	34,633	right preauricular area
   143	EEG C3-LE	12,818	left inner medial
   144	EEG C3-P3	2	the area between the left inner medial and the left inner posterior
   145	EEG C3P-REF	13,687	unknown
   146	EEG C3-REF	40,686	left inner medial
   147	EEG C3-T3	2	the area between the left inner medial and the left outer medial
   148	EEG C4-CZ	2	the area between the right inner medial and the center top of the scalp (the apex)
   149	EEG C4-LE	12,818	right inner medial
   150	EEG C4-P4	2	the area between the right inner medial and the right inner posterior
   151	EEG C4P-REF	13,684	unknown
   152	EEG C4-REF	40,686	right inner medial
   153	EEG CZ-C3	2	the area between the center top of the scalp (apex) and the left inner medial
   154	EEG CZ-LE	12,818	center top of the scalp (apex)
   155	EEG CZ-PZ	2	the area between CZ and PZ
   156	EEG CZ-REF	40,685	center top of the scalp (apex)
   157	EEG EKG1-REF	37,477	Electrocardiogram, single electrode on chest
   158	EEG EKG-LE	12,802	Electrocardiogram, single electrode on chest
   159	EEG EKG-REF	555	Electrocardiogram, single electrode on chest
   160	EEG F3-C3	2	the area between the left inner frontal and the left inner medial
   161	EEG F3-LE	12,818	inner frontal
   162	EEG F3-REF	40,685	inner frontal
   163	EEG F4-C4	2	the area between the right inner frontal and the right inner medial
   164	EEG F4-LE	12,818	right inner frontal
   165	EEG F4-REF	40,686	right inner frontal
   166	EEG F7-LE	12,818	left outer frontal
   167	EEG F7-REF	40,686	left outer frontal
   168	EEG F7-T3	2	the area between the left outer frontal and the left outer medial
   169	EEG F8-LE	12,818	right outer frontal
   170	EEG F8-REF	40,686	right outer frontal
   171	EEG F8-T4	2	the area between the right outer frontal and the right outer medial
   172	EEG FP1-F7	2	the area between the left forehead and the left outer frontal
   173	EEG FP1-LE	12,818	left forehead
   174	EEG FP1-REF	40,686	left forehead
   175	EEG FP2-F8	2	the area between the right forehead and the right outer frontal
   176	EEG FP2-LE	12,818	right forehead
   177	EEG FP2-REF	40,686	right forehead
   178	EEG FZ-CZ	2	the area between FZ and CZ
   179	EEG FZ-LE	12,818	halfway between the apex and the forehead
   180	EEG FZ-REF	40,685	halfway between the apex and the forehead
   181	EEG LOC-REF	17,566	unknown
   182	EEG LUC-LE	1,621	unknown
   183	EEG LUC-REF	523	unknown
   184	EEG O1-LE	12,818	left back of head
   185	EEG O1-REF	40,686	left back of head
   186	EEG O2-LE	12,818	right back of head
   187	EEG O2-REF	40,686	right back of head
   188	EEG OZ-LE	12,802	center back of head
   189	EEG OZ-REF	58	center back of head
   190	EEG P3-LE	12,818	left inner posterior
   191	EEG P3-REF	40,686	left inner posterior
   192	EEG P4-LE	12,818	right inner posterior
   193	EEG P4-REF	40,686	right inner posterior
   194	EEG PG1-LE	12,222	unknown
   195	EEG PG1-REF	15	unknown
   196	EEG PG2-LE	12,222	unknown
   197	EEG PG2-REF	15	unknown
   198	EEG PZ-LE	12,818	halfway between the apex and the back of the head
   199	EEG PZ-REF	40,685	halfway between the apex and the back of the head
   200	EEG RESP1-REF	523	on the body to record respiration
   201	EEG RESP2-REF	523	on the body to record respiration
   202	EEG RLC-LE	1,621	unknown
   203	EEG RLC-REF	523	unknown
   204	EEG ROC-REF	17,567	unknown
   205	EEG SP1-LE	3,921	left side of the head on the sphenoid (approximately the temple)
   206	EEG SP1-REF	14,906	left side of the head on the sphenoid (approximately the temple)
   207	EEG SP2-LE	3,921	right side of the head on the sphenoid (approximately the temple)
   208	EEG SP2-REF	14,238	right side of the head on the sphenoid (approximately the temple)
   209	EEG T1-LE	4,501	left side of the head on the sphenoid (approximately the temple)
   210	EEG T1-REF	37,952	left side of the head on the sphenoid (approximately the temple)
   211	EEG T1-T2	2	the area between T1 and T2
   212	EEG T2-LE	4,501	right side of the head on the sphenoid (approximately the temple)
   213	EEG T2-REF	37,956	right side of the head on the sphenoid (approximately the temple)
   214	EEG T2-T4	2	the area between T2 and T4
   215	EEG T3-LE	12,818	left outer medial
   216	EEG T3-REF	40,686	left outer medial
   217	EEG T3-T1	2	the area between T3 and T1
   218	EEG T3-T5	2	the area between the left outer medial and the left outer posterior
   219	EEG T4-C4	2	the area between the right outer medial and the right inner medial
   220	EEG T4-LE	12,818	right outer medial
   221	EEG T4-REF	40,686	right outer medial
   222	EEG T4-T6	2	the area between the right outer medial and the right outer posterior
   223	EEG T5-LE	12,818	left outer posterior
   224	EEG T5-O1	2	the area between the left outer posterior and the left back of head
   225	EEG T5-REF	40,686	left outer posterior
   226	EEG T6-LE	12,818	right outer posterior
   227	EEG T6-O2	2	the area between the right outer posterior and the right back of head
   228	EEG T6-REF	40,686	right outer posterior
   229	EEG X1-REF	41	custom electrode placement
   230	EMG-REF	17,931	A single electrode placed on a muscle belly 
   231	IBI	37,546	interburst intervals
   232	PHOTIC PH	12,222	photic stimulation
   233	PHOTIC-REF	14,632	photic stimulation
   234	PULSE RATE	70	a single ECG lead, cardiac activity
   235	RESP ABDOMEN-REF	944	a belt placed under the arms and across the chest
   236	SUPPR	37,546	unknown
Appendix C.	Channel Labels in DU (v1.0)
Index	Label	Freq	Description
1	A1	43	equivalent to EEG A1-REF
2	A2	43	equivalent to EEG A2-REF
3	C3	43	equivalent to EEG C3-REF
4	C4	43	equivalent to EEG C4-REF
5	CO2WAVE	43	 unknown
6	CZ	43	equivalent to EEG CZ-REF
7	DC01	2	DC1-DC
8	DC02	2	DC2-DC
9	DC03	45	DC3-DC
10	DC04	45	DC4-DC
11	DC05	45	DC5-DC
12	DC06	45	DC6-DC
13	DC07	2	DC7-DC
14	DC08	2	DC8-DC
15	E	43	ear electrode, usually on the earlobe and used for a reference
16	EDF ANNOTATIONS	45	annotations created by EEG technician
17	EEG A1	2	equivalent to EEG A1-REF
18	EEG A2	2	equivalent to EEG A2-REF
19	EEG C3	2	equivalent to EEG C3-REF
20	EEG C4	2	equivalent to EEG C4-REF
21	EEG CZ	2	equivalent to EEG CZ-REF
22	EEG E	2	ear electrode, usually on the earlobe and used for a reference
23	EEG F3	2	equivalent to EEG F3-REF
24	EEG F4	2	equivalent to EEG F4-REF
25	EEG F7	2	equivalent to EEG F7-REF
26	EEG F8	2	equivalent to EEG F8-REF
27	EEG FP1	2	equivalent to EEG FP1-REF
28	EEG FP2	2	equivalent to EEG FP2-REF
29	EEG FZ	2	equivalent to EEG FZ-REF
30	EEG MARK1	10	unknown
31	EEG MARK2	10	unknown
32	EEG O1	2	equivalent to EEG O1-REF
33	EEG O2	2	equivalent to EEG O2-REF
34	EEG P3	2	equivalent to EEG P3-REF
35	EEG P4	2	equivalent to EEG P3-REF
36	EEG PG1	2	unknown
37	EEG PG2	2	unknown
38	EEG PZ	2	equivalent to EEG PZ-REF
39	EEG T1	2	equivalent to EEG T1-REF
40	EEG T2	2	equivalent to EEG T2-REF
41	EEG T3	2	equivalent to EEG T3-REF
42	EEG T4	2	equivalent to EEG T4-REF
43	EEG T5	2	equivalent to EEG T5-REF
44	EEG T6	2	equivalent to EEG T6-REF
45	EEG X1	2	unknown
46	EEG X2	2	unknown
47	EEG X3	2	unknown
48	EEG X4	2	unknown
49	EEG X5	2	unknown
50	EEG X6	2	unknown
51	EEG X7	2	unknown
52	ETCO2	43	unknown
53	EVENTS/MARKERS	45	unknown
54	F3	43	equivalent to EEG F3-REF
55	F4	43	equivalent to EEG F4-REF
56	F7	43	equivalent to EEG F7-REF
57	F8	43	equivalent to EEG F8-REF
58	FP1	43	equivalent to EEG FP1-REF
59	FP2	43	equivalent to EEG FP2-REF
60	FZ	43	equivalent to EEG FZ-REF
61	O1	43	equivalent to EEG O1-REF
62	O2	43	equivalent to EEG O2-REF
63	P3	43	equivalent to EEG P3-REF
64	P4	43	equivalent to EEG P3-REF
65	PG1	43	unknown
66	PG2	43	unknown
67	PULSE	43	EKG-REF
68	PZ	43	equivalent to EEG PZ-REF
69	SPO2	43	unknown
70	T1	43	equivalent to EEG T1-REF
71	T2	43	equivalent to EEG T2-REF
72	T3	43	equivalent to EEG T3-REF
73	T4	43	equivalent to EEG T4-REF
74	T5	43	equivalent to EEG T5-REF
75	T6	43	equivalent to EEG T6-REF
76	X1	43	unknown
77	X2	43	unknown
78	X3	43	unknown
79	X4	43	unknown
80	X5	43	unknown
81	X6	43	unknown
81	X7	43	unknown

