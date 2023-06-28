import nibabel as nib
import numpy as np
import readCMRRPhysio_1 
import glob
import sys
import os 
from scipy import interpolate
from scipy.signal import filtfilt, resample
import matplotlib.pyplot as plt
import nipy.modalities.fmri.design_matrix as dm
from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.modalities.fmri.glm import FMRILinearModel
import logging

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

    # Öffnen der Nifti-Datei
    mrt_data = nib.load(nii_file_path).get_fdata()

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

    return mrt_data, physio

# 10924365 ist der erste aufgenommene Zeitpunkt, ohne ist Start bei 0!
# slicemap = Zeitpunkte der Aufnahme eines Volumens in 3D
# mrt_data = Werte eines Volumens in 3D
def time_stamps_to_image_conformation(slicemap: list, mrt_data: list):
    """
    Konvertiert Zeitstempelinformationen in die Konformation eines Bildes und verbindet sie mit den MRT-Daten.

    Parameter:
    - slicemap: Eine Liste oder ein Array mit den Zeitstempelinformationen.
    - mrt_data: Eine Liste oder ein Array mit den MRT-Daten.

    Rückgabewert:
    Ein 5-dimensionales Array, das die Konformation des Bildes enthält 
    und die Zeitstempelinformationen mit den MRT-Daten verbindet.
    all_data[x, y, slice, volume, [Wert, Startzeitpunkt der Aufnahme, Endzeitpunkt der Aufnahme]]
    """
    x, y, slice, volume = mrt_data.shape
    all_data = np.zeros((x, y, slice, volume, 3))
    # Über die Volumes und Slices die beiden Datensätze verbinden
    for sli in range(slice):
        for vol in range(volume):
            time_start = slicemap[0, vol, sli, 0]
            time_end = slicemap[1, vol, sli, 0]
            all_data[:, :, sli, vol, 0] = mrt_data[:, :, sli, vol]
            all_data[:, :, sli, vol, 1] = time_start
            all_data[:, :, sli, vol, 2] = time_end
    return all_data

# Spline Interpolation, um die Nullwerte zu entfernen und zu interpolieren
def spline_interpolation(data: list):
    """
    Führt eine Spline-Interpolation auf den gegebenen Daten aus, um Nullwerte zu entfernen und zu interpolieren.

    Parameter:
    - data: Eine Liste oder ein Array mit den Daten, auf die die Spline-Interpolation angewendet werden soll.

    Rückgabewert:
    Eine interpolierte Version der Daten als Liste oder Array.
    """
    data_new = data
    # Maske, die Werte <100 markiert
    mask = (data_new < 100)
    # Interpolationsfunktion
    spline_function = interpolate.interp1d(
                      np.flatnonzero(~mask), data_new[~mask], 
                      kind='cubic', fill_value="extrapolate")
    # Ersetzen Sie die Nullwerte durch Spline-Interpolation
    data_new[mask] = spline_function(np.flatnonzero(mask))
    return data_new

# Lowpass moving average Filter + Verschiebung eliminieren
def lowpass(data: list):
    """
    Führt einen Tiefpassfilter mit gleitendem Mittelwert auf den gegebenen Daten aus und eliminiert Verschiebungen.

    Parameter:
    - data: Eine Liste oder ein Array mit den Daten, auf die der Tiefpassfilter angewendet werden soll.

    Rückgabewert:
    Eine gefilterte Version der Daten als Liste oder Array.
    """
    window_size = 100
    b = np.ones(window_size)/window_size
    smooth_data = filtfilt(b, 1, data)
    smooth_data = smooth_data-np.mean(smooth_data)
    return smooth_data

# TODO Reicht das Downsampling aus?
def downsample(data: list):
    """
    Führt eine Abtastung auf den gegebenen Daten durch, um die Anzahl der Datenpunkte zu verringern.

    Parameter:
    - data: Eine Liste oder ein Array mit den Daten, die abgetastet werden sollen.

    Rückgabewert:
    Eine abgetastete Version der Daten als Liste oder Array.
    """
    # Durch /329 Sampling auf 420
    resampled_signal = resample(data, num=int(np.prod(data.shape)/329))
    return resampled_signal

def plotting_resp_puls(Respiration: list, Puls: list): 

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

    # Passt den Abstand zwischen den Plots an
    plt.tight_layout()

    # Zeigt das Ergebnis an
    plt.show()

def model_design_matrix(design: list):
    #TODO
    tr = 0.8 # repetition time is 0.8 second
    n_scans = 420  # the acquisition comprises 420*72 scans
    frametimes = np.arange(n_scans) * tr  # here are the corresponding frame times

    # TODO Wie erstelle ich die Designmatrix auf Grundlage der 

    design_matrix = dm.make_dmtx(frametimes, paradigm=None, 
                                 add_regs=[puls_prepro], add_reg_names=['Pulse'])
    
    return design_matrix
    
def model_glm(mrt_data: list, design_matrix: list):
    # TODO Wie modelliere ich richtig für unsere Daten? Wie kann ich die 
    # unterschiedlichen Timestamps mit implemeniteren?
    cval = np.hstack((1, np.zeros(9))) # TODO Kontrastwerte
    model = GeneralLinearModel(mrt_data)
    model.fit(design_matrix)
    z_vals = model.contrast(cval).z_score() # Je höher je besser, bei p Werten je niedriger je besser

    return 0

# Eingabeaufforderung für den Benutzer, um den Dateipfad einzugeben
folder_path = input("Bitte geben Sie den Dateipfad zum Datensazu ein: ")
mrt_data, physio = search_data(folder_path)

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

all_data = time_stamps_to_image_conformation(physioSliceMap, mrt_data)

puls_prepro = downsample(lowpass(spline_interpolation(Puls)))

resp_prepro = downsample(lowpass(spline_interpolation(Respiration)))

design_matrix = model_design_matrix(puls_prepro)

# Designmatrix anzeigen
design_matrix.show()

# Optional: Titel und Achsenbeschriftungen hinzufügen
plt.title('Designmatrix')
plt.xlabel('Zeit')
plt.ylabel('Regressor-Werte')

# Das Ergebnis anzeigen
plt.show()

model_glm(all_data, design_matrix)