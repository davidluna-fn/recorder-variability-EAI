# Tratamiento de datos
# ==============================================================================
from typing import overload
import warnings

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.mlab as plb
from matplotlib import style
import seaborn as sns
from scipy import stats
import soundfile as sf
from scipy import signal, stats
from pydub import AudioSegment, effects
import math
import time
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.patches as patches
from matplotlib import transforms


# Configuración matplotlib
# ==============================================================================
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
warnings.filterwarnings('ignore')

# Varios
# ==============================================================================

import numpy as np
import soundfile as sf
import pandas as pd
import pathlib
from pathlib import Path
import librosa
from tqdm.notebook import tqdm
from datetime import datetime

from Filtros import filtro_hpf, filtro_lpf

from indices import (ACIft,
                     ADI,
                     beta,
                     temporal_entropy,
                     median_envelope,
                     NDSI,
                     number_of_peaks)

def audioread(path,name_format='BLACKA_%Y%m%d_%H%M%S.wav'):
    path = Path(path)
    files = list(path.glob("*.{}".format('[wW][aA][vV]')))
    if isinstance(path, pathlib.WindowsPath):
        names = list( map(lambda x: str(x).split('\\')[-1] , files) )
    else:
        names = list( map(lambda x: str(x).split('/')[-1] , files) )
        
    dates = list( map(lambda x: datetime.strptime(x,name_format) , names) )
    x,fs = sf.read(files[0])
    len_audio = np.floor((len(x)/fs))
    
    fs_ = np.repeat(fs,len(files))
    len_ = np.repeat(len_audio,len(files))
    
    data = np.vstack((names,dates,fs_,len_)).T
    df = pd.DataFrame(columns=['name','date','fs','length_seg'], data=data)
    return df

def get_pdf(data1,data2,number,xlim=None):
    kernel1 = stats.gaussian_kde(data1)
    kernel2 = stats.gaussian_kde(data2)
    if xlim == None:
        x = np.linspace(np.min(np.hstack((data1,data2))) - np.std(np.hstack((data1,data2))), 
            np.max(np.hstack((data1,data2)))+ np.std(np.hstack((data1,data2))), number )
    else:
        x = np.linspace(xlim[0], xlim[1], number )
    pdf1 = kernel1.pdf(x)
    pdf1 /= np.sum(pdf1)
    pdf2 = kernel2.pdf(x)
    pdf2 /= np.sum(pdf2)
    return pdf1,pdf2, x


def index_temporal(df : pd.DataFrame, index_: str):
    mean_ = df.groupby(df['date'].dt.date).mean()[index_].values
    std_  = df.groupby(df['date'].dt.date).std()[index_].values
    date_ = df.groupby(df['date'].dt.date).mean()[index_].index
    return mean_,std_, date_

def load_audios(path, df):
    audios = []
    for x in tqdm(df.index.values):
        x, Fs = sf.read(f'{path}/{df.loc[x]["name"]}')
        audios.append(x)
    return audios


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_metrics(name, v1,v2,p,q):
    corr = np.round(np.corrcoef(v1,v2)[0,1],2)
    kld  = np.round(kl_divergence(p, q) + kl_divergence(q, p),2)/2
    _, p_value = stats.mannwhitneyu(p,q)
    p_value = np.round(1 - p_value,5)
    return [name,corr,kld,p_value]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtaifrom scipy import stats
n the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def get_datetime(df,col,prefix,extension):
    df['datetime'] = df[col].str.replace(prefix, '')
    df['datetime'] = df['datetime'].str.replace(extension, '')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d_%H%M%S')
    return df

def get_dataplot(df,col1,col2,group = 'hour'):
    if group == 'hour':
        datay = df.groupby(df[col1].dt.hour).sum()[col2].values
        datax = df.groupby(df[col1].dt.hour).mean()[col2].index
    elif group == 'day':
        datay = df.groupby(df[col1].dt.day).sum()[col2].values
        datax = df.groupby(df[col1].dt.day).mean()[col2].index
    elif group == 'month':
        datay = df.groupby(df[col1].dt.month).sum()[col2].values
        datax = df.groupby(df[col1].dt.month).mean()[col2].index
    return datax, datay


def get_indices(name, processed=False, fs2=None):
    try:
        s, fs = torchaudio.load(str(name))
        if s.size()[0] != 1:
            s = torch.unsqueeze(s[0,:], 0)
        
        if fs != fs2 and processed == True:
            s = F.resample(s,fs,fs2,resampling_method="kaiser_window")
            fs = fs2
        
        indc = []
        tamano_ventana = 512
        nfft = 512
        tipo_ventana ="hann",
        sobreposicion = 0

        if processed == True:
            s = s/torch.sqrt(torch.mean(s**2))
            Sxx, tn, fn, ext = maad.sound.spectrogram(s[0,:], fs, mode='amplitude')
            Sxx_power,_,_,_ = maad.sound.spectrogram(s[0,:], fs)

            ACIft_ = indices.ACIft(Sxx)
            ADI  = maad.features.acoustic_diversity_index(Sxx,fn,fmax=12000)
            BETA = maad.features.bioacoustics_index(Sxx,fn)
            M = maad.features.temporal_median(s[0,:])
            NP = maad.features.number_of_peaks(Sxx_power, fn, slopes=6, min_freq_dist=100, display=False)

            Hf, _ = maad.features.frequency_entropy(Sxx_power) 
            Ht = maad.features.temporal_entropy(s.numpy()[0,:])
            H = Ht*Hf
            AEI = maad.features.acoustic_eveness_index(Sxx,fn, fmax=12000)
            NDSI, _, _, _  = maad.features.soundscape_index(Sxx_power,fn)
            indc.extend([str(name).split('/')[-1],str(name).split('/')[-2],ACIft_,ADI,BETA,M,NP,H,AEI,NDSI])

            return indc

        else:
            Sxx, tn, fn, ext = maad.sound.spectrogram(s[0,:], fs, mode='amplitude')
            Sxx_power,_,_,_ = maad.sound.spectrogram(s[0,:], fs)


            ACIft_ = indices.ACIft(Sxx)
            ADI  = maad.features.acoustic_diversity_index(Sxx,fn,fmax=int(fs/2))
            BETA = maad.features.bioacoustics_index(Sxx,fn)
            M = maad.features.temporal_median(s[0,:])
            NP = maad.features.number_of_peaks(Sxx_power, fn, slopes=6, min_freq_dist=100, display=False)
            Hf, _ = maad.features.frequency_entropy(Sxx_power) 
            Ht = maad.features.temporal_entropy(s.numpy()[0,:])
            H = Ht*Hf
            AEI = maad.features.acoustic_eveness_index(Sxx,fn,fmax=int(fs/2))
            NDSI, _, _, _  = maad.features.soundscape_index(Sxx_power,fn)

            indc.extend([str(name).split('/')[-1],str(name).split('/')[-2],ACIft_,ADI,BETA,M,NP,H,AEI,NDSI])

            return indc

    except:
        return np.nan



def get_psd(name, processed=False, fs2=None):
    try:
        s, fs = torchaudio.load(str(name))
        if s.size()[0] != 1:
            s = torch.unsqueeze(s[0,:], 0)
        
        if fs != fs2 and processed == True:
            s = F.resample(s,fs,fs2,resampling_method="kaiser_window")
            fs = fs2
                
        if fs == 48000:
            freq, Pxx = welch(s, fs, nfft=557, noverlap=0)
        elif fs == 44100:
            freq, Pxx = welch(s, fs, nfft=512, noverlap=0)

        return [str(name).split('/')[-1], str(name).split('/')[-2], Pxx, freq]

    except:
        return np.nan


def get_spectrogram(name, processed=False, fs2=None):
    try:
        s, fs = torchaudio.load(str(name))
        if s.size()[0] != 1:
            s = torch.unsqueeze(s[0,:], 0)
        
        if fs != fs2 and processed == True:
            s = F.resample(s,fs,fs2,resampling_method="kaiser_window")
            fs = fs2
                

        f, t, Sxx = spectrogram(x=s, fs=fs,nfft=512, scaling='spectrum',mode ='magnitude', noverlap=0)
        return [str(name).split('/')[-1], str(name).split('/')[-2], Sxx, t, f]

    except:
        return np.nan