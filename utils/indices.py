'''
Contiene los algoritmos que calculan cada uno de los posibles descriptores de paisaje acústico.

'''

import math
import numpy as np
from scipy import signal, stats

def ACItf(audio, Fs, j, s):

    '''

    Calcula el indice de complejidad acústica original (ACI) [1] que fue renombrado en [2]

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz (int)
    :param j: tamaño de cada cluster en segundos (int)
    :param s: espectrograma de la señal (numpy array)
    :return: valor del ACItf (float)
    '''

    #Para comparar grabaciones con diferente duración, hacer ACItf/t
    s = s/np.amax(s)
    specrows = s.shape[0]
    speccols = s.shape[1]

    dk = np.absolute(np.diff(s, axis=1)) #length speccols - 1

    #duracion de la grabación
    duracion = len(audio)/Fs
    m = math.ceil(duracion/j) #Número de j en total
    n = math.floor(speccols/m) #Número de tk en j

    ACI = np.zeros((specrows, m))

    for t in range(0, m):
        k1 = range(n*t, n*(t+1)-1)
        k2 = range(n*t, n*(t+1))
        D = np.sum(dk[:, k1], axis=1)
        ACI[:, t] = np.divide(D, np.sum(s[:, k2], axis = 1))

    ACI_dft = np.sum(ACI, axis=0)
    ACItot = sum(ACI_dft)

    return ACItot

def ACIft(s):

    '''

    Calcula el índice de complejidad acústica que se calcula primero sobre la frecuencia y luego sobre el tiempo (ACIft)
    [2]

    :param s: Espectrograma de la señal (numpy array)
    :return: el valor del ACIft (float)
    '''

    s = s / np.amax(s)
    ACI = np.sum(np.divide(np.absolute(np.diff(s, axis=0)), s[1:, :] + s[:-1, :]))
    return ACI

def ADI(s, Fmax = 10000, wband = 1000, bn = -50):

    '''

    Calcula el índice de diversidad acústica (ADI) descrito en [3]

    :param s: Espectrograma de la señal (numpy array)
    :param Fmax: Frecuencia máxima para el análisis en Hz, valor por defecto 10000 (int)
    :param wband: tamaño de cada banda de frecuencia en Hz, valor por defecto 1000 (int)
    :param bn: Valor del umbral (ruido de fondo) en dBFS, valor por defecto -50 (int)
    :return: Valor del ADI (float)
    '''

    s = s/np.amax(s)
    bn = 10**(bn/20)
    sclean = s - bn
    sclean[sclean < 0] = 0
    sclean[sclean != 0] = 1
    nband = int(Fmax//wband)
    bin_step = int(s.shape[0]//nband)
    pbin = np.sum(sclean, axis=1)/s[:bin_step,:].size
    p = np.zeros(nband)

    for band in range(nband):
        p[band] = np.sum(pbin[band*bin_step:(band+1)*bin_step]) + 0.0000001

    ADIv = -np.multiply(p, np.log(p))
    ADItot = np.sum(ADIv)
    return ADItot

def ADIm(s, Fs, wband=1000):

    '''
    Calcula el vector de ADI modificado propuesto en [4]

    :param s: Espectrograma de la señal (numpy array)
    :param Fs: Frecuencia de muestreo en Hz (int)
    :param wband: tamaño de cada banda de frecuencia en Hz, valor por defecto 1000 (int)
    :return: Un vector que contiene los valores del ADIm (numpy array)
    '''

    bn = background_noise_freq(s)
    #bn=-50
    #bn = 10**(bn/20)
    sclean = s - np.tile(bn, (s.shape[1], 1)).T
    sclean[sclean < 0] = 0
    sclean[sclean != 0] = 1
    Fmax = Fs/2
    nband = int(Fmax//wband)
    bin_step = int(s.shape[0]//nband)
    pbin = np.sum(sclean, axis=1)/s[:bin_step,:].size
    p = np.zeros(nband)

    for band in range(nband):
        p[band] = np.sum(pbin[band*bin_step:(band+1)*bin_step]) + 0.0000001

    ADIv = -np.multiply(p, np.log(p))
    return ADIv

def background_noise_freq(s):

    '''

    Calcula el valor del ruido de fondo para cada celda del espectrograma en el eje de las frecuencias [5]

    :param s: Espectrograma de la señal (numpy array)
    :return: Vector que contiene el valor del ruido de fondo para cada celda de frecuencia (numpy array)
    '''

    nfbins = s.shape[0]
    bn = np.zeros(nfbins)
    for i in range(nfbins):
        f = s[i, :]
        nbins = int(s.shape[1]/8)
        H, bin_edges = np.histogram(f, bins=nbins)
        fwin = 5
        nbinsn = H.size-fwin
        sH = np.zeros(nbinsn)

        for j in range(nbinsn):
            sH[j] = H[j:j+fwin].sum()/fwin

        modep = sH.argmax()
        mode = np.amin(f) + (np.amax(f)-np.amin(f))*(modep/nbins)

        acum = 0
        j = 0
        Hmax = np.amax(sH)
        while acum < 0.68*Hmax:
            acum += H[j]
            j += 1

        nsd = np.amin(f) + (np.amax(f)-np.amin(f))*(j/nbins)
        bn[i] = mode + 0.1*nsd
    return bn

def background_noise_time(SPL, fwin):

    '''

    Calcula el valor del ruido de fondo de la señal en el tiempo [5]

    :param SPL: Señal con el nivel de presión sonora (SPL) de la señal en dB (numpy array)
    :param fwin: Tamaño de la ventana temporal para el análisis (int)
    :return: el valor de ruido de fondo en dB (float)
    '''

    SPLmin = min(SPL)
    HdB, bin_edges = np.histogram(SPL, range=(SPLmin, SPLmin + 10))
    sHdB = np.zeros((len(HdB)-fwin, 1))

    for i in range(len(HdB) - fwin):
        sHdB[i] = sum(HdB[i:i+fwin]/fwin)

    modep = np.argmax(sHdB)
    bn = SPLmin + 0.1*modep
    return bn

def beta(s, f, bio_band=(2000, 8000)):

    '''

    Calcula el índice bioacústico de la señal (β) [6]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param bio_band: tupla con la frecuencia mínima y máxima de la banda biofónica, valor por defecto: (2000, 8000) (tuple)
    :return: el valor de β (float)
    '''

    minf = bio_band[0]
    maxf = bio_band[1]
    s = s/np.amax(s)
    s = 10*np.log10(np.mean(s**2, axis=1))
    bioph = s[np.logical_and(f>=minf, f<= maxf)]
    bioph_norm = bioph - np.amin(bioph, axis=0)
    B = np.trapz(bioph_norm, f[np.logical_and(f>=minf, f<= maxf)])
    return B

def crest_factor(audio, rms):

    '''

    El factor de creasta es el cociente entre el valor pico de la señal de energía (Epeak) y su valor cuadrático medio
    (RMS). Los valores altos indican muchos picos en la señal de energía. [7]

    :param audio: señal monoaural temporal (numpy array)
    :param rms: valor RMS de la señal (float)
    :return: retorna el factor de cresta de la señal (float)
    '''

    audio2 = audio ** 2
    mint = max(audio2)
    cf = mint/rms
    return cf

def frequency_modulation(s):

    '''

    La modulación frecuencial es el ángulo medio de las derivadas direccionales. Los valores altos indican cambios
    abruptos en la intensidad. [8]

    :param s: Espectrograma de la señal (numpy array)
    :return: valor de la modulación frecuencial
    '''

    ds_df = np.diff(s, axis=0)
    ds_dt = np.diff(s, axis=1)
    fm = np.mean(np.absolute(np.arctan(np.divide(-ds_df[:, 1:], -ds_dt[1:, :])))*(180/math.pi))
    return fm

def meanspec(audio, Fs= 1, wn="hann", ovlp=0, wl=512, nfft = None, norm=True):

    '''

    Calcula el espectro medio haciendo el promedio en el eje de las frecuencias del espectrograma.

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz, valor por defecto 1 (int)
    :param wn: tipo de ventana, valor por defecto "hann" (str)
    :param ovlp: puntos de solapamiento entre ventanas, valor por defecto 0 (int)
    :param wl: tamaño de la ventana, valor por defecto 512 (int)
    :param nfft: número de puntos de la transformada de Fourier, valor por defecto, None, es decir el mismo de wl (int)
    :param norm: booleano que indica si se normaliza o no el espectro, valor por defecto, True.
    :return: señal con el espectro medio (numpy array)
    '''

    f, t, Zxx = signal.stft(audio, fs = Fs, window=wn, noverlap=ovlp, nperseg=wl, nfft=nfft)
    mspec = np.mean(np.abs(Zxx), axis=1)

    if norm == True:
        mspec = mspec/max(mspec)

    return f, mspec

def median_envelope(audio, Fs, depth=16):

    '''

    La mediana del envolvente de la amplitud (M)[9].

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz (int)
    :param depth: la profundidad de digitalización de la señal, valor por defecto 16 (int)
    :return: el valor de M (float)
    '''

    min_points = Fs*60
    npoints = len(audio)
    y = []
    VerParticion = npoints/min_points

    if(VerParticion >= 3 ):
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))
    else:
        if(VerParticion==1):
            min_points = Fs*20
        else:
            min_points = Fs*30
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))

    y = np.concatenate([y])
    M = (2**(1-depth))*np.median(y)
    return M

def mid_band_activity(s, f, fmin = 450, fmax = 3500):

    '''

    Calcula la actividad acústica en la banda media [10]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param fmin: frecuencia inferior de la banda media en Hz, valor por defecto 450 (int)
    :param fmax: frecuencia superior de la banda media en Hz, valor por defecto 3500 (int)
    :return: valor de la actividad acústica en la banda media (float)
    '''

    s = np.mean(s, axis=1)
    s = s**2
    s = s/np.amax(s)
    threshold = 10*np.log10(np.mean(s))
    s = 10 * np.log10(s)
    MID = np.sum(s[np.logical_and(f>=fmin, f<= fmax)]>threshold)/len(s)
    return MID

def musicality_degree(audio, Fs, win=256, nfft=None, type_win="hann", overlap=None):

    '''

    La pendiente media de la curva 1/f [11]

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz (int)
    :param win: tamaño de la ventana, valor por defecto 256 (int)
    :param nfft: número de puntos de la transformada de Fourier, valor por defecto, None, es decir el mismo de win (int)
    :param type_win: tipo de ventana, valor por defecto "hann" (str)
    :param overlap: puntos de solapamiento entre ventanas, valor por defecto None, es decir win/2 (float)
    :return: valor del grado de musicalidad (float)
    '''

    f, pxx = signal.welch(audio, Fs, nperseg=win, nfft=nfft, window=type_win, noverlap=overlap)
    f += 0.0000001
    lp2 = np.log10(pxx ** 2)
    lf = np.log10(f)
    dlf = np.diff(lf)
    dlp2 = np.diff(lp2)
    md_v = np.divide(dlp2, dlf)
    md = np.mean(md_v)
    return md

def NDSI(s, f, bio_band = (2000, 8000), tech_band = (200, 1500)):

    '''

    Calcula el índice NDSI [12] que hace una relación entre el nivel de biofonía y tecnofonía. -1 indica biofonía pura y
    1 indica pura tecnofonía.

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param bio_band:  tupla con la frecuencia mínima y máxima de la banda biofónica, valor por defecto: (2000, 8000) (tuple)
    :param tech_band: tupla con la frecuencia mínima y máxima de la banda tecnofónica, valor por defecto: (200, 1500) (tuple)
    :return: el valor NDSI de la señal (float)
    '''
    s = np.mean(s, axis=1)
    s = s ** 2

    bio = s[np.logical_and(f>=bio_band[0], f<=bio_band[1])]
    B = np.trapz(bio, f[np.logical_and(f>=bio_band[0], f<=bio_band[1])])

    tech = s[np.logical_and(f>=tech_band[0], f<=tech_band[1])]
    A = np.trapz(tech, f[np.logical_and(f>=tech_band[0], f<=tech_band[1])])

    ND = (B-A)/(B+A)
    return ND

def number_of_peaks(s, f, nedges=10):

    '''

    Cuenta el número de picos en el espectro medio de la señal [13].

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param nedges: número de partes en las que se divide la señal, por defecto 10 (int)
    :return: número de picos de la señal.
    '''

    #Filtro de media móvil
    def smooth(a, n=10):

        '''

        Esta función suaviza la señal con un filtro de media móvil.

        :param a: señal (numpy array)
        :param n: tamaño de la ventana del filtro de media móvil, por defecto, 10 (int)
        :return: señal suavizada
        '''

        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    s = np.sum(s, axis=1)
    s = s/np.amax(s)
    s = 20*np.log10(s)
    s = smooth(smooth(s)) #suavizo la señal
    f = smooth(smooth(f))
    s -= np.amin(s)
    ds = s[1:] - s[:-1]
    df = f[1:] - f[:-1]
    dsdf = np.divide(ds, df)

    step = round(len(s)/nedges)
    meansig = [np.mean(np.abs(s[j*step:(j+1)*step])) for j in range(nedges)]

    ind = []

    if s[0] > meansig[0] and s[0] > 1.2*np.mean(s) and np.mean(dsdf[1:4])<0:
        ind.append(0)

    for i in range(4, len(s)-3):
        if s[i] > meansig[i%nedges] and s[i] > 1.2*np.mean(s) and np.mean(dsdf[i+1:i+4])>0 and np.mean(dsdf[i-4:i-1]) < 0:
            ind.append(i)

    if s[-1] > meansig[0] and s[-1] > 1.2*np.mean(s) and np.mean(dsdf[-4:-1]) > 0:
        ind.append(len(s)-1)

    if len(ind) == 0:
        NP = 0
        return NP

    NP = 1
    df_p = f[ind[1:]] - f[ind[:-1]]
    acum = 0

    for i in df_p:
        acum += i
        if acum >= 200:
            NP += 1
            acum = 0

    return NP

def rho(s, f, bio_band = (2000, 8000), tech_band = (200, 1500)):

    '''

    Razón entre biofonía y tecnofonía (ρ) [14]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param bio_band:  tupla con la frecuencia mínima y máxima de la banda biofónica, valor por defecto: (2000, 8000) (tuple)
    :param tech_band: tupla con la frecuencia mínima y máxima de la banda tecnofónica, valor por defecto: (200, 1500) (tuple)
    :return: valor de ρ (float)
    '''

    s = np.mean(s, axis=1)
    s = s ** 2

    bio = s[np.logical_and(f>=bio_band[0], f<=bio_band[1])]
    B = np.trapz(bio, f[np.logical_and(f>=bio_band[0], f<=bio_band[1])])

    tech = s[np.logical_and(f>=tech_band[0], f<=tech_band[1])]
    A = np.trapz(tech, f[np.logical_and(f>=tech_band[0], f<=tech_band[1])])

    P = B/A
    return P

def rms(audio):

    '''

    Calcula el valor RMS de la señal

    :param audio: señal monoaural temporal (numpy array)
    :return: valor RMS
    '''

    erms = math.sqrt(sum(audio ** 2))
    return erms

def spectral_maxima_entropy(s, f, fmin, fmax):

    '''

    Calcula la entropía de los máximos espectrales (Hm)[10]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param fmin: frecuencia inferior de la banda en la que se hará el análisis en Hz (int)
    :param fmax: frecuencia superior de la banda en la que se hará el análisis en Hz (int)
    :return: valor del Hm (float)
    '''

    s = s/np.amax(s)
    s_max = np.max(s, axis=1)
    s_band = s_max[np.logical_and(f >= fmin, f>=fmax)]
    s_norm = s_band/np.sum(s_band)
    N = len(s_norm)
    Hm = -np.sum(np.multiply(s_norm, np.log2(s_norm)))/np.log2(N)
    return Hm

def spectral_variance_entropy(s, f, fmin, fmax):

    '''

     Calcula la entropía de la varianza espectral (Hv)[10]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param fmin: frecuencia inferior de la banda en la que se hará el análisis en Hz (int)
    :param fmax: frecuencia superior de la banda en la que se hará el análisis en Hz (int)
    :return: valor del Hv (float)
    '''

    s = s/np.amax(s)
    s_std = np.std(s, axis=1)
    s_band = s_std[np.logical_and(f >= fmin, f>=fmax)]
    s_norm = s_band/np.sum(s_band)
    N = len(s_norm)
    Hv = -np.sum(np.multiply(s_norm, np.log2(s_norm)))/np.log2(N)
    return Hv

def temporal_entropy(audio, Fs):

    '''

    Calcula la entropía acústica temporal (Ht)[15]

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz (int)
    :return: el valor de Ht (float)
    '''

    min_points = Fs*60
    npoints = len(audio)
    y = []
    VerParticion = npoints/min_points

    if(VerParticion >= 3 ):
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))
    else:
        if(VerParticion==1):
            min_points = Fs*20
        else:
            min_points = Fs*30
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))

    env = np.concatenate([y])
    env_norm = env/np.sum(env)

    N = len(env_norm)
    Ht = -np.sum(np.multiply(env_norm, np.log2(env_norm)))/np.log2(N)
    return Ht

def wav2SPL(audio, sen, gain, Vrms):

    '''

    Calcula el nivel de presión sonora de la señal [16]

    :param audio: señal monoaural temporal (numpy array)
    :param sen: sensibilidad del micrófono en dB (float)
    :param gain: ganancia del micrófono en dB (float)
    :param Vrms: Voltaje RMS del conversor análogo digital de la grabadora (float)
    :return: señal del nivel de presión sonora (numpy array)
    '''

    audio += 2 ** -17
    Vp = Vrms*math.sqrt(2)
    S = sen + gain + 20*math.log10(1/Vp)
    SPL = 20 * np.log10(np.absolute(audio)) - S
    return SPL

def wiener_entropy(audio, win=256, nfft=None, type_win="hann", overlap=None):

    '''

    Calcula la entropia de wiener o spectral flatness (SF) [17]

    :param audio: señal monoaural temporal (numpy array)
    :param win: tamaño de la ventana, valor por defecto 256 (int)
    :param nfft: número de puntos de la transformada de Fourier, valor por defecto, None, es decir el mismo de win (int)
    :param type_win: tipo de ventana, valor por defecto "hann" (str)
    :param overlap: puntos de solapamiento entre ventanas, valor por defecto None, es decir win/2 (float)
    :return: el valor de la entropia de wiener (float)
    '''

    f, pxx = signal.welch(audio, nperseg=win, nfft=nfft, window=type_win, noverlap=overlap)
    num = stats.mstats.gmean(pxx)
    den = np.mean(pxx)
    spf = num/den
    return spf

def signaltonoise(a, axis=0, ddof=0):

    '''
    Calcula la relación señal a ruido (SNR)
    :param audio: señal monoaural temporal (numpy array)
    :param axis: Si el eje es igual a Ninguno, la matriz se desdobla primero, valor predeterminado = 0
    :param ddof: Corrección de los grados de libertad para la desviación estándar. El valor predeterminado = 0.
    '''
    mx = np.amax(a)
    a = np.divide(a,mx)
    a = np.square(a)
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis)
    snr = np.where(sd == 0, 0, m/sd)
    return snr


    '''


    Referencias:

    [1] Pieretti, N., Farina, A., & Morri, D. (2011). A new methodology to infer the singing activity of an avian
        community: The Acoustic Complexity Index (ACI). Ecological Indicators, 11(3), 868–873.
        http://doi.org/10.1016/j.ecolind.2010.11.005

    [2] Farina, A., Pieretti, N., Salutari, P., Tognari, E., & Lombardi, A. (2016). The Application of the Acoustic
        Complexity Indices (ACI) to Ecoacoustic Event Detection and Identification (EEDI) Modeling. Biosemiotics, 9(2),
        227–246. http://doi.org/10.1007/s12304-016-9266-3

    [3] Pekin, B. K., Jung, J., Villanueva-Rivera, L. J., Pijanowski, B. C., & Ahumada, J. A. (2012). Modeling acoustic
        diversity using soundscape recordings and LIDAR-derived metrics of vertical forest structure in a neotropical
        rainforest. Landscape Ecology, 27(10), 1513–1522. http://doi.org/10.1007/s10980-012-9806-4

    [4] Duque-Montoya, D. C. (2018). Methodology for Ecosystem Change Assessing using Ecoacoustics Analysis.
        Universidad de Antioquia.

    [5] Towsey, M. (2013). Noise removal from waveforms and spectrograms derived from natural recordings of the
        environment. Retrieved from http://eprints.qut.edu.au/61399/

    [6] Boelman, N. T., Asner, G. P., Hart, P. J., & Martin, R. E. (2007). Multi-trophic invasion resistance in Hawaii:
        Bioacoustics, field surveys, and airborne remote sensing. Ecological Applications, 17(8), 2137–2144.
        http://doi.org/10.1890/07-0004.1

    [7] Torija, A. J., Ruiz, D. P., & Ramos-Ridao, a F. (2013). Application of a methodology for categorizing and
        differentiating urban soundscapes using acoustical descriptors and semantic-differential attributes.
        The Journal of the Acoustical Society of America, 134(1), 791–802. http://doi.org/10.1121/1.4807804

    [8] Tchernichovski, O., Nottebohm, F., Ho, C., Pesaran, B., & Mitra, P. (2000). A procedure for an automated
        measurement of song similarity. Animal Behaviour, 59(6), 1167–1176. http://doi.org/10.1006/anbe.1999.1416

    [9] Depraetere, M., Pavoine, S., Jiguet, F., Gasc, A., Duvail, S., & Sueur, J. (2012). Monitoring animal diversity
        using acoustic indices: Implementation in a temperate woodland. Ecological Indicators, 13(1), 46–54.
        http://doi.org/10.1016/j.ecolind.2011.05.006

    [10] Towsey, M., Wimmer, J., Williamson, I., & Roe, P. (2014). The use of acoustic indices to determine avian
        species richness in audio-recordings of the environment. Ecological Informatics, 21, 110–119.
        http://doi.org/10.1016/j.ecoinf.2013.11.007

    [11] De Coensel, B., Botteldooren, D., Debacq, K., Nilsson, M. E., & Berglund, B. (2007).
        Soundscape classifying ants. In Internoise. http://doi.org/10.1260/135101007781447993

    [12] Kasten, E. P., Gage, S. H., Fox, J., & Joo, W. (2012). The remote environmental assessment laboratory’s
        acoustic library: An archive for studying soundscape ecology. Ecological Informatics, 12, 50–67.
        http://doi.org/10.1016/j.ecoinf.2012.08.001

    [13] Gasc, A., Sueur, J., Pavoine, S., Pellens, R., & Grandcolas, P. (2013). Biodiversity Sampling Using a Global
        Acoustic Approach: Contrasting Sites with Microendemics in New Caledonia. PLoS ONE, 8(5), e65311.
        http://doi.org/10.1371/journal.pone.0065311

    [14] Qi, J., Gage, S. H., Joo, W., Napoletano, B., & Biswas, S. (2007). Soundscape characteristics of an
        environment: a new ecological indicator of ecosystem health. In Wetland and Water Resource Modeling and
        Assessment: A Watershed Perspective (Vol. 20071553, pp. 201–214). http://doi.org/10.1201/9781420064155

    [15] Sueur, J., Pavoine, S., Hamerlynck, O., & Duvail, S. (2008). Rapid Acoustic Survey for Biodiversity Appraisal.
        PLoS ONE, 3(12), e4065. http://doi.org/10.1371/journal.pone.0004065

    [16] Merchant, N. D., Fristrup, K. M., Johnson, M. P., Tyack, P. L., Witt, M. J., Blondel, P., & Parks, S. E.
        (2015). Measuring acoustic habitats. Methods in Ecology and Evolution, 6(3), 257–265.
        http://doi.org/10.1111/2041-210X.12330

    [17] Mitrović, D., Zeppelzauer, M., & Breiteneder, C. (2010). Features for Content-Based Audio Retrieval.
         Advances in Computers, 78(10), 71–150. http://doi.org/10.1016/S0065-2458(10)78003-7
    '''

