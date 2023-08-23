import nibabel as nib
import numpy as np
import readCMRRPhysio_1 
import glob
import sys
import os 
from scipy import interpolate
from scipy.signal import filtfilt
import scipy.stats as stats
import matplotlib.pyplot as plt
import nipy.modalities.fmri.design_matrix as dm
import nipy.modalities.fmri.glm as glm
from nipy.labs.viz_tools.activation_maps import demo_plot_map, plot_anat, plot_map 
import logging
from matplotlib.colors import LinearSegmentedColormap

LOGGER = logging.getLogger('Fallseminar')

def search_data(folder_path):
    # Suche nach der Nifti-Datei
    search_pattern_nifti = "*.nii"
    nii_files = glob.glob(folder_path + search_pattern_nifti)

    # Überprüfen, ob genau eine NIFTI-Datei gefunden wurde
    if len(nii_files) != 1:
        print("Fehler: Es wurde nicht genau eine NIfTI-Datei gefunden.")
        sys.exit()

    # Wenn es nur eine gibt, wird weiter fortgefahren
    nii_file_path = nii_files[0]
    print("Folgende NIfTI-Datei wurde gefunden:", nii_file_path)
    # Wenn es nur eine gibt, wird weiter fortgefahren
    nii_file_path = nii_files[0]
    print("Folgende NIfTI-Datei wurde gefunden:", nii_file_path)

    # Öffnen der Nifti-Datei
    mrt_data = nib.load(nii_file_path).get_fdata()
    mrt_data_nib = nib.load(nii_file_path)

    # Suche nach der Info Datei (egal ob RESP/PULS/Info, der Rest sollte gleich sein)
    search_pattern_physio = "*_Info.log"
    physio_files = glob.glob(folder_path + search_pattern_physio)

    # Überprüfen, ob genau eine _Info.log-Datei gefunden wurde
    if len(physio_files) != 1:
        print("Fehler: Es wurde nicht genau eine _Info.log-Datei gefunden.")
        sys.exit()

    # Wenn es nur eine gibt, wird weiter fortgefahren
    physio_file_path = physio_files[0]
    print("Folgende _Info.log-Datei wurde gefunden:", physio_file_path)

    # Entfernen der Endung von _Info.log für readPhysio()
    physio_file_name = os.path.splitext(os.path.basename(physio_file_path))[0]
    physio_file_name = folder_path +physio_file_name.split("_Info")[0]

    # readPhysio aufrufen
    physio = readCMRRPhysio_1.readCMRRPhysio(physio_file_name)

    return mrt_data, mrt_data_nib, physio

# 10924365 ist der erste aufgenommene Zeitpunkt, ohne ist Start bei 0!
# slicemap = Zeitpunkte der Aufnahme eines Volumens in 3D

# 10924365 ist der erste aufgenommene Zeitpunkt, ohne ist Start bei 0!
# slicemap = Zeitpunkte der Aufnahme eines Volumens in 3D
# mrt_data = Werte eines Volumens in 3D
def time_stamps_to_image_conformation(slicemap: list, mrt_data: list):
    x, y, slice, volume = mrt_data.shape
    all_data = np.zeros((x, y, slice, volume, 3))
    # Über die Volumes und Slices die beiden Datensätze verbinden
    for sli in range(slice):
        for vol in range(volume):
            time_start = slicemap[0, vol, sli, 0]
            time_end = slicemap[1, vol, sli, 0]
            all_data[:, :, sli, vol, 0] = mrt_data[:, :, sli, vol]
            all_data[:, :, sli, vol, 1] = (time_start - time_end)/2
            all_data[:, :, sli, vol, 2] = time_end
    return all_data

# Spline Interpolation, um die Nullwerte zu entfernen und zu interpolieren
def spline_interpolation(data: list):
    data_new = data
    # Maske, die Werte <100 markiert
    mask = (data_new < 100)
    # Interpolationsfunktion
    # TODO Finite Impulse Response Filter
    spline_function = interpolate.interp1d(
                      np.flatnonzero(~mask), data_new[~mask], 
                      kind='cubic', fill_value="extrapolate")
    # Ersetzen Sie die Nullwerte durch Spline-Interpolation
    data_new[mask] = spline_function(np.flatnonzero(mask))
    return data_new

#  Lowpass moving average Filter + Verschiebung eliminieren
def lowpass(data: list):
    # TODO Samplefrequenz der Daten Betrachten, Samplefrequenz 420 Volumina/345 sec 
    # Scandauer = 1,24
    window_size = 100
    b = np.ones(window_size)/window_size
    smooth_data = filtfilt(b, 1, data)
    smooth_data = smooth_data-np.mean(smooth_data)
    return smooth_data

def downsample(slicemap: list, slice, delay, physiologic_data):
    resample_physio = np.zeros(420)
    for i in range(420):
        time_start = slicemap[0, i, slice]
        time_end = slicemap[1, i, slice]
        time_mw = int((time_end+time_start)/2)
        resample_physio[i] = physiologic_data[time_mw+delay]

    return resample_physio

def plotting_resp_puls(Respiration: list, Puls: list): 

    # Erstellen Sie den 2x4 Plotsatz
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
    # Erstellen Sie den 2x4 Plotsatz
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    # Data Plot
    axs[0,0].plot(Respiration)
    axs[0,0].set_title('RESP Raw')
    Respiration_inter = spline_interpolation(Respiration)
    axs[0,1].plot(Respiration_inter)
    axs[0,1].set_title('RESP Spline interpolated')
    Respiration_inter_lowpass = lowpass(Respiration_inter)
    axs[0,2].plot(Respiration_inter_lowpass)
    axs[0,2].set_title('RESP Lowpass')
    Respiration_inter_lowpass_resample = downsample(Respiration_inter_lowpass)
    axs[0,3].plot(Respiration_inter_lowpass_resample)
    axs[0,3].set_title('RESP Downsample')

    axs[1,0].plot(Puls)
    axs[1,0].set_title('PULS Raw')
    Puls_inter = spline_interpolation(Puls)
    axs[1,1].plot(Puls_inter)
    axs[1,1].set_title('PULS Spline interpolated')
    Puls_inter_lowpass = lowpass(Puls_inter)
    axs[1,2].plot(Puls_inter_lowpass)
    axs[1,2].set_title('PULS Lowpass')
    Puls_inter_lowpass_resample = downsample(Puls_inter_lowpass)
    axs[1,3].plot(Puls_inter_lowpass_resample)
    axs[1,3].set_title('PULS Downsample')
    # Data Plot
    axs[0,0].plot(Respiration)
    axs[0,0].set_title('RESP Raw')
    Respiration_inter = spline_interpolation(Respiration)
    axs[0,1].plot(Respiration_inter)
    axs[0,1].set_title('RESP Spline interpolated')
    Respiration_inter_lowpass = lowpass(Respiration_inter)
    axs[0,2].plot(Respiration_inter_lowpass)
    axs[0,2].set_title('RESP Lowpass')
    Respiration_inter_lowpass_resample = downsample(Respiration_inter_lowpass)
    axs[0,3].plot(Respiration_inter_lowpass_resample)
    axs[0,3].set_title('RESP Downsample')

    axs[1,0].plot(Puls)
    axs[1,0].set_title('PULS Raw')
    Puls_inter = spline_interpolation(Puls)
    axs[1,1].plot(Puls_inter)
    axs[1,1].set_title('PULS Spline interpolated')
    Puls_inter_lowpass = lowpass(Puls_inter)
    axs[1,2].plot(Puls_inter_lowpass)
    axs[1,2].set_title('PULS Lowpass')
    Puls_inter_lowpass_resample = downsample(Puls_inter_lowpass)
    axs[1,3].plot(Puls_inter_lowpass_resample)
    axs[1,3].set_title('PULS Downsample')

    # Passt den Abstand zwischen den Plots an
    plt.tight_layout()
    # Passt den Abstand zwischen den Plots an
    plt.tight_layout()

    # Zeigt das Ergebnis an
    plt.show()
    # Zeigt das Ergebnis an
    plt.show()

def model_design_matrix(design: list):
    #TODO
    tr = 0.8 # repetition time is 0.8 second
    n_scans = 420  # the acquisition comprises 420*72 scans
    frametimes = np.arange(n_scans) *tr  # here are the corresponding frame times

    # TODO Wie erstelle ich die Designmatrix auf Grundlage der 

    design_matrix = dm.dmtx_light(frametimes, paradigm=None, 
                                 add_regs=design, add_reg_names=['Pulse'])
    
    return design_matrix
    
def model_glm(mrt_data, design_matrix):
    # TODO Wie modelliere ich richtig für unsere Daten? Wie kann ich die 
    # unterschiedlichen Timestamps mit implemeniteren?
    model = glm.FMRILinearModel(mrt_data, design_matrix)
    model.fit()
    # Je höher je besser, bei p Werten je niedriger je besser
    z_image, effect_image = model.contrast(np.ones(6), output_effects = True)

    image_data = z_image.get_fdata()
    p_value = 0.001
    significance_threshold = stats.norm.ppf(1 - p_value/2)
    print(significance_threshold)
    colors = [(1, 1, 0), (1, 0, 0)]
    activation_cmap = LinearSegmentedColormap.from_list('activation_cmap', colors, N=256)
    thresholded_activation_map = np.where(image_data[:,:,36] >= significance_threshold, image_data[:,:,36], np.nan)

    background_data = mrt_data.get_fdata()

    plt.imshow(background_data[:,:,36,15], cmap='gray')
    plt.imshow(thresholded_activation_map, cmap=activation_cmap, alpha=0.7)
    plt.axis('off')
    plt.show()
    return 0

# Eingabeaufforderung für den Benutzer, um den Dateipfad einzugeben
folder_path = input("Bitte geben Sie den Dateipfad zum Datensazu ein: ")
mrt_data, mrt_data_nib, physio = search_data(folder_path)

physioUUID = physio['UUID']    
physioFreq = physio['Freq']
physioSliceMap = physio['SliceMap']
physioACQ = physio['ACQ']
physioFirsttime = physio['firsttime']
Respiration = physio['RESP']
Puls = physio['PULS']
LOGGER.info(f"Respiration time ticks:      {Respiration.shape}")
LOGGER.info(f"Puls time ticks:      {Puls.shape}")
LOGGER.info(f"FMRI Data time ticks:      {mrt_data.shape}")


puls_prepro = lowpass(spline_interpolation(Puls))

downsamled_data = downsample(physioSliceMap, 36, 0, puls_prepro)

resp_prepro = lowpass(spline_interpolation(Respiration))

design_matrix = model_design_matrix(downsamled_data)

# Designmatrix anzeigen
#design_matrix.show()
#print(design_matrix[0].shape)
#print(mrt_data[:,:,36, :].shape)
# Optional: Titel und Achsenbeschriftungen hinzufügen
#plt.title('Designmatrix')
#plt.xlabel('Zeit')
#plt.ylabel('Regressor-Werte')

# Das Ergebnis anzeigen
#plt.show()

model_glm(mrt_data_nib, design_matrix[0])