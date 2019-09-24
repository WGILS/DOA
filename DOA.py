from scipy.signal import fftconvolve
import numpy as np
from scipy import stats
from scipy.io import wavfile
import csv
import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist

######
# We define a meaningful distance measure on the circle

# Location of original 
azimuth = 61. / 180. * np.pi  # 60 degrees
distance = 3.  # 3 meters

#######################
# algorithms parameters
SNR = 0.    # signal-to-noise ratio
c = 343.    # speed of sound
fs = 44100  # sampling frequency
nfft = 256  # FFT size
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# We use a circular array with radius 5 cm and 6 microphones
R = pra.circular_2D_array((5,5), 6, 0., 0.0463)

#Reading the data from a wav file with 6 channels - Should be changed for stream data
fs, data = wavfile.read('audio_1min.wav')

readings = np.zeros([5])
max_samples = 10
ti=int(fs/5)
time=0

# The output will be a CSV file.  Could also be a json file
writeFile= open('doa.csv', 'w')
writer = csv.writer(writeFile)
row=['frame','doa']
writer.writerow(row)

print('Audio file loaded')
print('Audio length:',data.shape[0]/fs,'seconds')
print('Number of blocks: ',data.shape[0] // ti)

# For loop to create the data in the shape needed by the DoA algorithm
for blocks in range(0,data.shape[0] // ti):
    signals=[]
    for i in range(0,data.shape[1]):
        signals.append(data[blocks*ti:(blocks+1)*ti,i])
    X = np.array([
        pra.stft(signal, nfft, nfft // 2, transform=np.fft.rfft).T
        for signal in signals ])


    algo_name="SRP"

    # Construct the new DOA object
    doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)

    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_bins=freq_bins)
    row=[blocks,doa.azimuth_recon / np.pi * 180.]
    writer.writerow(row)
    if blocks%100==0:
        print(blocks/(data.shape[0] // ti)*100,"%")

# Close the CSV file
writeFile.close()
