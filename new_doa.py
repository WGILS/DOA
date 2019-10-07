import soundfile as sf
import numpy as np

data1,fs=sf.read('./audio_6channels22-01.WAV')
data2,fs=sf.read('./audio_6channels22-02.WAV')
data3,fs=sf.read('./audio_6channels22-03.WAV')
data4,fs=sf.read('./audio_6channels22-04.WAV')
data5,fs=sf.read('./audio_6channels22-05.WAV')
data6,fs=sf.read('./audio_6channels22-06.WAV')

#data = np.zeros((data1.shape[0],6))
data = np.stack((data1[:,0],data2[:,0],data3,data4,data5,data6),axis=1)


filename = 'audio22_6.wav'
sf.write(filename, data, fs)


readings = np.zeros([5])
max_samples = 10
ti=int(fs/5)#designate 0.2s as one block?
time=0



import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist

######
# We define a meaningful distance measure on the circle

# Location of original source
azimuth = 61. / 180. * np.pi  # 60 degrees
distance = 3.  # 3 meters

#######################
# algorithms parameters
SNR = 0.    # signal-to-noise ratio
c = 343.    # speed of sound
fs = 44100  # sampling frequency
nfft = 256  # FFT size
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# We use a circular array with radius 15 cm # and 12 microphones
R = pra.circular_2D_array((5,5), 6, 0., 0.0463)
source = np.array([2, 5])






doatxt=""
for blocks in range(0,data.shape[0] // ti):
#for blocks in range(0,10):
    signals=[]
    for i in range(0,data.shape[1]):
        signals.append(data[blocks*ti:(blocks+1)*ti,i])
    #print(len(signals[0]))
    X = np.array([ 
        pra.stft(signal, nfft, nfft // 2, transform=np.fft.rfft).T 
        for signal in signals ])
    #X shape should be (6,129,24)
    
    algo_name="SRP"   
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
    doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)  
    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_bins=freq_bins)
    
    doatxt += str(blocks*0.2)+" "+str(doa.azimuth_recon[0] / np.pi * 180.)+"\n"
    if blocks%100==0:
        #this is the percentage of being done
        print(blocks/(data.shape[0] // ti)*100,"%")


f= open("./doa22.txt","w+")
f.write(doatxt)
f.close()


