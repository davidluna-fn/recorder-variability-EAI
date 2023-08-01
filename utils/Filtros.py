'''Filtros pasabanda, pasabaja y pasa alta
   implementado febrero de 2021
   autor: Jessica Loaiza'''

import math
import numpy as np
from scipy import signal, stats
from scipy import signal as sn
import scipy.signal as sp
from scipy.fft import fftshift

def filtro_lpf(fc,audio,fs):
    fc=fc+150
    Adb=0 # Ganancia del filtro en decibeles
    N = 512 # Numero de puntos de la FFT
    BW=50 # Ancho de banda en la banda de tansicion

    wc = 2*np.pi*fc/fs # frecuencia de corte normalizada en radianes
    bwn=2*np.pi*BW/fs # ancho de banda  normalizado en radianes

    M=int(4/bwn) # orden estimado del filtro
    n = np.arange(-M,M)
    #Diseño filtro pasa baja
    h1 = wc/np.pi * np.sinc(wc*(n)/np.pi) # Respuesta del filtro ideal 
    w1,Hh1 = signal.freqz(h1,1,whole=True, worN=N) # Respuesta en frecuencia del filtro ideal

    win= signal.hamming(len(n)) # funcion ventana para eliminar el fenomeno de Gibbs

    h2=h1*win # Multiplico la respuesta ideal por la ventana

    A=np.sqrt(10**(0.1*Adb))
    h2=h2*A # Ganancia del filtro
    w2,Hh2 = signal.freqz(h2,1,whole=True, worN=N) # Respuesta en frecuencia del filtro enventanado

    sg_filtrada = sp.lfilter(h2, 1, audio)
    return sg_filtrada

def filtro_hpf(fc,audio,fs):
    fc=fc+150
    Adb=0 # Ganancia del filtro en decibeles
    N = 512 # Numero de puntos de la FFT
    BW=50 # Ancho de banda en la banda de tansicion

    wc = 2*np.pi*fc/fs # frecuencia de corte normalizada en radianes
    bwn=2*np.pi*BW/fs # ancho de banda  normalizado en radianes

    M=int(4/bwn) # orden estimado del filtro
    n = np.arange(-M,M)
    #Diseño filtro pasa alta
    h1 = -wc/np.pi * np.sinc(wc*(n)/np.pi) # Respuesta del filtro ideal 
    h1[n==0]=1-(wc)/np.pi #truncar respuesta en el origen
    w1,Hh1 = signal.freqz(h1,1,whole=True, worN=N) # Respuesta en frecuencia del filtro ideal

    win= signal.hamming(len(n)) # funcion ventana para eliminar el fenomeno de Gibbs

    h2=h1*win # Multiplico la respuesta ideal por la ventana

    A=np.sqrt(10**(0.1*Adb))
    h2=h2*A # Ganancia del filtro
    w2,Hh2 = signal.freqz(h2,1,whole=True, worN=N) # Respuesta en frecuencia del filtro enventanado

    sg_filtrada = sp.lfilter(h2, 1, audio)
    return sg_filtrada

def filtro_bpf(fcl,fch,audio,fs):
    fch=fch+150
    fcl=fcl+150
    Adb=0 # Ganancia del filtro en decibeles
    N = 512 # Numero de puntos de la FFT
    BW=50 # Ancho de banda en la banda de tansicion

    wch = 2*np.pi*fch/fs # frecuencia de corte normalizada en radianes
    wcl = 2*np.pi*fcl/fs
    bwn=2*np.pi*BW/fs # ancho de banda  normalizado en radianes

    M=int(4/bwn) # orden estimado del filtro
    n = np.arange(-M,M)
    #Diseño filtro pasa banda
    h1 = wch/np.pi * np.sinc(wch*(n)/np.pi)-wcl/np.pi * np.sinc(wcl*(n)/np.pi) # Respuesta del filtro ideal 
    h1[n==0]=(wch-wcl)/np.pi #truncar respuesta en el origen para pasabanda

    w1,Hh1 = signal.freqz(h1,1,whole=True, worN=N) # Respuesta en frecuencia del filtro ideal

    win= signal.hamming(len(n)) # funcion ventana para eliminar el fenomeno de Gibbs

    h2=h1*win # Multiplico la respuesta ideal por la ventana

    A=np.sqrt(10**(0.1*Adb))
    h2=h2*A # Ganancia del filtro
    w2,Hh2 = signal.freqz(h2,1,whole=True, worN=N) # Respuesta en frecuencia del filtro enventanado

    sg_filtrada = sp.lfilter(h2, 1, audio)
    return sg_filtrada