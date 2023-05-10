import nibabel as nib
import numpy as np
import readCMRRPhysio_1 
import glob
import sys
import os 
from scipy import interpolate
from scipy.signal import filtfilt, resample
import matplotlib.pyplot as plt

# Eingabeaufforderung für den Benutzer, um den Dateipfad einzugeben
folder_path = input("Bitte geben Sie den Dateipfad zum Datensazu ein: ")

#Überprüfen, ob der Dateipfad gültig ist. Gültigkeit erfolgt, wenn der Dateiüfad wiefolgt angegeben wurde:
#"Testdaten/20220809/run2/"
if os.path.exists(folder_path):
    print("Der angegebene Pfad ist gültig.")
else:
    print("Der angegebene Pfad ist ungültig.")
    sys.exit()

#Suche nach der Nifti-Datei
search_pattern_nifti = "*.nii"
nii_files = glob.glob(folder_path + search_pattern_nifti)

#Überprüfen, ob genau eine NIFTI-Datei gefunden wurde
if len(nii_files) > 1:
    print("Fehler: Es wurde nicht genau eine NIfTI-Datei gefunden.")
    sys.exit()
if len(nii_files) < 1:
    print("Fehler: Es wurde keine NIfTI-Datei gefunden.")
    sys.exit()

#Wenn es nur eine gibt, wird weiter fortgefahren
nii_file_path = nii_files[0]
print("Folgende NIfTI-Datei wurde gefunden:", nii_file_path)

#Öffnen der Nifti-Datei
mrt_data = nib.load(nii_file_path).get_fdata()

#Suche nach der Info Datei (egal ob RESP/PULS/Info, der Rest sollte gleich sein)
search_pattern_physio = "*_Info.log"
physio_files = glob.glob(folder_path + search_pattern_physio)

#Überprüfen, ob genau eine _Info.log-Datei gefunden wurde
if len(physio_files) != 1:
    print("Fehler: Es wurde nicht genau eine _Info.log-Datei gefunden.")
    sys.exit()

#Wenn es nur eine gibt, wird weiter fortgefahren
physio_file_path = physio_files[0]
print("Folgende _Info.log-Datei wurde gefunden:", physio_file_path)

#Entfernen der Endung von _Info.log fpr readPhysio()
physio_file_name = os.path.splitext(os.path.basename(physio_file_path))[0]
physio_file_name = folder_path +physio_file_name.split("_Info")[0]

#readPhysio aufrufen
physio = readCMRRPhysio_1.readCMRRPhysio(physio_file_name)
physioUUID = physio['UUID']    
physioFreq = physio['Freq']
physioSliceMap = physio['SliceMap']
physioACQ = physio['ACQ']
physioFirsttime = physio['firsttime']
Respiration = physio['RESP']
Puls = physio['PULS']

#10924365 ist der erste aufgenommene Zeitpunkt, ohne ist Start bei 0!
#slicemap = Zeitpunkte der Aufnahme eines Volumens in 3D
#mrt_data = Werte eines Volumens in 3D
def time_stamps_to_image_conformation(slicemap, mrt_data, firsttime):
    x, y, slice, volume = mrt_data.shape
    all_data = np.zeros((x, y, slice, volume, 3))
    #Über die Volumes und Slices die beiden Datensätze verbinden
    for sli in range(slice):
        for vol in range(volume):
            time_start = firsttime + slicemap[0, vol, sli, 0]
            time_end = firsttime + slicemap[1, vol, sli, 0]
            all_data[:, :, sli, vol, 0] = mrt_data[:, :, sli, vol]
            all_data[:, :, sli, vol, 1] = time_start
            all_data[:, :, sli, vol, 2] = time_end
    #Rückgabe von all_data mit den Informationen:
    #all_data[x, y, slice, volume, 
    #[Wert, Startzeitpunkt der Aufnahme, Endzeitpunkt der Aufnahme]]
    return all_data

#Spline Interpolation, um die Nullwerte zu entfernen und zu interpolieren
def spline_interpolation(data):
    data_new = data
    #Maske, die Werte <100 markiert
    mask = (data_new < 100)
    #Interpolationsfunktion
    spline_function = interpolate.interp1d(
                      np.flatnonzero(~mask), data_new[~mask], 
                      kind='cubic', fill_value="extrapolate")
    # Ersetzen Sie die Nullwerte durch Spline-Interpolation
    data_new[mask] = spline_function(np.flatnonzero(mask))
    return data_new

#Lowpass moving average Filter + Verschiebung eliminieren
def lowpass(data):
    window_size = 100
    b = np.ones(window_size)/window_size
    smooth_data = filtfilt(b, 1, data)
    smooth_data = smooth_data-np.mean(smooth_data)
    return smooth_data

def downsample(data):
    resampled_signal = resample(data, num=int(np.prod(data.shape)/329))
    return resampled_signal

# Erstellen Sie den 2x4 Plotsatz
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

# Data Plot
axs[0,0].plot(Respiration)
axs[0,0].set_title('RESP Raw')
Respiration = spline_interpolation(Respiration)
axs[0,1].plot(Respiration)
axs[0,1].set_title('RESP Spline interpolated')
Respiration = lowpass(Respiration)
axs[0,2].plot(Respiration)
axs[0,2].set_title('RESP Lowpass')
Respiration = downsample(Respiration)
axs[0,3].plot(Respiration)
axs[0,3].set_title('RESP Downsample')
axs[1,0].plot(Puls)
axs[1,0].set_title('PULS Raw')
Puls = spline_interpolation(Puls)
axs[1,1].plot(Puls)
axs[1,1].set_title('PULS Spline interpolated')
Puls = lowpass(Puls)
axs[1,2].plot(Puls)
axs[1,2].set_title('PULS Lowpass')
Puls = downsample(Puls)
axs[1,3].plot(Puls)
axs[1,3].set_title('PULS Downsample')

# Passt den Abstand zwischen den Plots an
plt.tight_layout()

# Zeigt das Ergebnis an
plt.show()
