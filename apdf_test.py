import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import SubplotParams
from novainstrumentation import smooth, filter
from matplotlib import rc
import os


def plot_tools():
    rc('lines', linewidth=2)

    rc('font', **{'family': 'serif', 'serif': ['Lato']})
    rc('font', size=25)

    rc('axes', grid=True, labelsize=20, titlesize=30)
    rc('grid', color='grey')
    rc('xtick', labelsize=20)
    rc('ytick', labelsize=20)
    rc('grid', linestyle='dotted')

    plt.close('all')


#function that converts raw data into millivolts
def transfer_functionEMG(emg_signal):
	n = 16
	vcc = 3
	g = 1000
	emg_mV = 1000*(((emg_signal/2**n)-1/2)*vcc)/g

	return emg_mV

#function that calculates mvc
def calculateMVC(dir_MVC, channel):
    #load file
	emg_sig_raw = np.loadtxt(dir_MVC)[:, channel]
    #change from raw to mV
	emg_sig_mV = transfer_functionEMG(emg_sig_raw)
    #filter the emg_signal
	emg_sig = filter.bandpass(emg_sig_mV, 20, 490, fs=1000)

    #calcualte mvc
	MVC = float(max(abs(emg_sig)))

	return MVC, emg_sig_raw


#function that calculates the apdf curve and the histogram distribution
def calculateApdf(emg):
    # envelope
    emg_env = smooth(abs(root_signal(emg)), window_len=50)

    # histogram
    histEmg, bins = np.histogram(emg_env, bins=100)

    # cumsum of hist
    apdf = np.cumsum(histEmg)

    return apdf, histEmg

#function that removes the DC component from the signal
def root_signal(sig):
	root_sig = sig - np.mean(sig)
	return root_sig





file_MVC = "data/MVC_Miguel/mvc_miguel_bicept_esquerda_000780FC5723_2018-07-09_13-25-10.txt"
file_1 = "data/MIGUEL/miguel_estacao40_esquerdo_futura_000780FC5723_2018-07-09_14-00-18.txt"


mvc, raw_mvc_signal = calculateMVC(file_MVC, 2)
bicept_channel = 5

emg = np.loadtxt(file_1)[:, bicept_channel]
emg_mV = transfer_functionEMG(emg)
emg_norm = emg_mV/mvc
emg_filtered = filter.bandpass(emg_norm, 20, 490)

apdf, hist = calculateApdf(emg_filtered)


#---------------------plot instructions---------------------------
#get apdf in percentage value
apdf = 100 * (apdf / max(apdf))

#time array in seconds
time = np.linspace(0, len(emg)/1000, len(emg))

# color
face_color_r = 248 / 255.0
face_color_g = 247 / 255.0
face_color_b = 249 / 255.0

# pars
left = 0.05  # the left side of the subplots of the figure
right = 0.95  # the right side of the subplots of the figure
bottom = 0.05  # the bottom of the subplots of the figure
top = 0.92  # the top of the subplots of the figure
wspace = 0.2  # the amount of width reserved for blank space between subplots
hspace = 0.6  # the amount of height reserved for white space between subplots

pars = SubplotParams(left, bottom, right, top, wspace, hspace)

n = 25
with sns.axes_style("whitegrid"):
    plot_tools()
    fig = plt.figure(figsize=(22, 14), facecolor=(face_color_r, face_color_g, face_color_b), dpi=50,
                     subplotpars=pars)
    ax1 = plt.subplot(411)
    ax1.patch.set_facecolor('ivory')
    ax1.plot(time, emg_filtered, color='darkslategray', linewidth=1.5)
    ax1.set_title('EMG raw')
    ax1.set_ylabel('%MVC', size=n)
    ax1.set_xlabel('Time (s)', size=n)
    ax1.axis('tight')
    ax1.legend()

    ax2 = plt.subplot(412)
    ax2.patch.set_facecolor('ivory')
    ax2.plot(time, smooth(abs(emg_filtered), window_len=50), color='darkslategray', linewidth=1.5)
    ax2.set_title('EMG Envelope')
    ax2.set_ylabel('%MVC', size=n)
    ax2.set_xlabel('Time (s)', size=n)
    ax2.axis('tight')
    ax2.legend()

    ax3 = plt.subplot(413)
    ax3.patch.set_facecolor('ivory')
    ax3.hist(smooth(abs(root_signal(emg_filtered)), window_len=50), bins=500, color='darkslategray')
    ax3.set_title('Histogram')
    ax3.set_ylabel('frequency (counts)', size=n)
    ax3.set_xlabel('%MVC', size=n)
    ax3.axis('tight')
    ax3.legend()

    ax4 = plt.subplot(414)
    ax4.patch.set_facecolor('ivory')
    ax4.plot(apdf, color='darkslategray', linewidth=5)
    ax4.set_title('Apdf curve')
    ax4.set_ylabel('P (%)', size=n)
    ax4.set_xlabel('%MVC', size=n)
    ax4.legend()

    plt.show()
