import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
from matplotlib import  transforms

from matplotlib import rcParams
rcParams["text.usetex"] = True
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelcolor"] = "black"
rcParams["figure.constrained_layout.use"] = True
rcParams["axes.titlesize"] = 22
rcParams["axes.facecolor"] = "white"
rcParams["axes.labelsize"] = 21
rcParams["legend.facecolor"] = "white"
rcParams["legend.edgecolor"] = "black"
rcParams['font.family'] = 'serif'

def format_ticks(ax, threshold=1000):
    """
    Formatea los ticks de una gráfica para mostrar números con una "k" si superan el límite especificado.
    
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): Eje de la gráfica.
        threshold (int): Límite para mostrar los ticks con una "k". Por defecto es 1000.
    """
    def formatter(x, pos):
        if x >= threshold:
            return f'{np.round(x/1000,2)}k'
        else:
            return x
    for axis in [ax.xaxis]:
        axis.set_major_formatter(ticker.FuncFormatter(formatter))
        
        
        
def graph_pdf(query_indices,tabla1,tabla2,point, db, show=True):
    db.execute(query_indices.format(tabla1=tabla1,tabla2=tabla2,folder=f'{point}A'))
    df1 = db.fetchall()

    db.execute(query_indices.format(tabla1=tabla1,tabla2=tabla2,folder=f'{point}S'))
    df2 = db.fetchall()

    indices = list(df1.columns[2:])

    pdf = PDF() 
    fig, ax = pdf.plot_pdf_indices(cols=indices, df1=df1, df2=df2, show=False)
    ax[0].set_title('ACIft')
    ax[2].set_title(r'$\beta$')
    if show:
        fig.show()
    return fig


class PDF:
    """
    Clase para generar y graficar las distribuciones de probabilidad de densidad de dos conjuntos de datos.
    """
    def __init__(self):
        """
        Inicializa los atributos de la clase.
        """
        
    def get_pdf(self, data1, data2, number, xlim=None):
        """
        Genera las distribuciones de probabilidad de densidad para los conjuntos de datos 1 y 2.
        Args:
            data1 (array-like): Conjunto de datos 1.
            data2 (array-like): Conjunto de datos 2.
            number (int): Número de puntos en el eje x para generar las distribuciones de probabilidad.
            xlim (tuple): Limites del eje x para generar las distribuciones de probabilidad. Por defecto es None.
        
        Returns:
            tuple: tupla con las distribuciones de probabilidad de densidad para los conjuntos de datos 1 y 2, y los valores del eje x.
        """
        kernel1 = stats.gaussian_kde(data1)
        kernel2 = stats.gaussian_kde(data2)
        x = np.linspace(
            np.min(np.hstack((data1, data2))) - np.std(np.hstack((data1, data2))), 
            np.max(np.hstack((data1, data2)))+ np.std(np.hstack((data1, data2))), 
            number) if xlim is None else np.linspace(xlim[0], xlim[1], number)
        
        pdf1 = kernel1.pdf(x)
        pdf2 = kernel2.pdf(x)
        return pdf1 / pdf1.sum(), pdf2 / pdf2.sum(), x

    def plot_pdf_indices(self, cols, df1, df2, df1label='Audiomoth', df2label='SM4', points=100, colors=['#009473','#FF6F61'],show=False):
        """
        Genera una gráfica de las distribuciones de probabilidad de densidad para cada columna especificada en cols.
        
        Args:
            cols (list): Lista de columnas para graficar.
            df1 (DataFrame): DataFrame con los datos para el conjunto de datos 1.
            df2 (DataFrame): DataFrame con los datos para el conjunto de datos 2.
            df1label (str): Etiqueta para el conjunto de datos 1 en la leyenda de la gráfica.
            df2label (str): Etiqueta para el conjunto de datos 2 en la leyenda de la gráfica.
            points (int): Número de puntos en el eje x para generar las distribuciones de probabilidad.
            colors (list): Lista de colores para graficar cada distribución de probabilidad.
        Returns:
        tuple: tupla con los objetos `figure` y `axis` generados por matplotlib.pyplot.
        """
        
        fig, axs = plt.subplots(2, 4, figsize=(14, 7))
        ax = axs.ravel()
        for i, index in enumerate(cols):
            pdf1, pdf2, x = self.get_pdf(df1[index],df2[index],points,xlim=None)

            ax[i].plot(x, pdf1, label=df1label, color=colors[0], linewidth = 2)
            l1 = ax[i].fill_between(x, np.zeros(len(x)), pdf1, color=colors[0], alpha=0.5, label=df1label)

            ax[i].plot(x, pdf2, label=df2label, color=colors[1], linewidth = 2)
            l2 = ax[i].fill_between(x, np.zeros(len(x)), pdf2, color=colors[1], alpha=0.5, label=df2label)
            
            ax[i].set_title(index.upper())
            ax[i].set_box_aspect(1)
            format_ticks(ax[i])
            ax[i].xaxis.set_major_locator(ticker.LinearLocator(3))
            ax[i].yaxis.set_major_locator(ticker.LinearLocator(4))
            ax[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax[i].spines[:].set_color("black")
            ax[i].set_ylim(0.00,None)
            ax[i].set_facecolor("white")
                
            
            if np.max(x) >= 1000:
                format_ticks(ax[i])

            if i >= 4:
                ax[i].set_xlabel("Index Value")

        ax[0].set_ylabel("Probability")
        ax[4].set_ylabel("Probability")
        labels = [df1label, df2label]
        fig.legend((l1, l2), labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels),fontsize=22)
        
        if show:
            fig.show()
    
        return fig, ax



def plot_psd_comparison(df1, df2):
    
    d1 = np.array([ 20*np.log10(np.array(x)/np.array(x).max()) for x in df1.psd.values ])
    d2 = np.array([ 20*np.log10(np.array(x)/np.array(x).max()) for x in df2.psd.values ])

    psd1 = np.mean(d1,axis=0)
    psd2 = np.mean(d2,axis=0)

    std1 = np.std(d1,axis=0)
    std2 = np.std(d2,axis=0)

    freq1 = np.array(df1.frequency.iloc[0])
    freq2 = np.array(df2.frequency.iloc[0])

    fig = plt.figure(figsize=(14,6))

    liminf1 = psd1 - std1
    limsup1 = psd1 + std1

    plt.plot(freq1,psd1,linewidth=3,label ='Audiomoth',color='#009473')
    l1 = plt.fill_between(freq1, liminf1,limsup1, color='#009473', alpha=0.4)

    liminf2 = psd2 - std2
    limsup2 = psd2 + std2

    plt.plot(freq2,psd2,linewidth=3,label = 'SM4', color="#FF6F61")
    l2 = plt.fill_between(freq2, liminf2,limsup2, color='#FF6F61', alpha=0.4)

    labels = ["Audiomoth", "SM4"]
    fig.legend((l1,l2),labels, loc='lower center', bbox_to_anchor=(0.25,0.16), ncol=len(labels), fontsize=22)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [dB/Hz]")


    plt.plot([22050, 22050],[10,-160], linewidth=2, color='gray')
    plt.plot([24000, 24000],[10,-160], linewidth=2, color='gray')
    plt.plot([10050, 10050],[10,-160], linewidth=2, color='gray')
    plt.plot([0, 24000],[-50,-50], linewidth=3, color='black')


    # intervalo 1

    plt.fill_between(freq1[np.logical_and(freq1 >= 10000, freq1 <= 22050)],
                    limsup1[np.logical_and(freq1 >= 10000, freq1 <= 22050)] + 0.3,
                    np.zeros(freq1[np.logical_and(freq1 >= 10000, freq1 <= 22050)].shape), 
                    color='blue', alpha=0.2)


    plt.fill_between(freq2[np.logical_and(freq2 >= 10000, freq2 <= 22050)], 
                    liminf2[np.logical_and(freq2 >= 10000, freq2 <= 22050)]-0.3,
                    -np.ones(freq2[np.logical_and(freq2 >= 10000, freq2 <= 22050)].shape)*160, 
                    color='blue', alpha=0.2)

    # medio 

    print(len(freq1[np.logical_and(freq1 >= 13200, freq1 <= 22050)]),len(limsup2[np.logical_and(freq2 >= 13200, freq2 <= 22050)]))
    plt.fill_between(freq1[np.logical_and(freq1 >= 13900, freq1 <= 22050)], 
                    liminf1[np.logical_and(freq1 >= 13900, freq1 <= 22050)]-0.3,
                    limsup2[np.logical_and(freq2 >= 13900, freq2 <= 22050)][1:]+0.9, 
                    color='blue', alpha=0.2)




    # intervalo 2

    plt.fill_between(freq1[np.logical_and(freq1 >= 22050, freq1 <= 24000)], 
                    limsup1[np.logical_and(freq1 >= 22050, freq1 <= 24000)] + 0.3,
                    np.zeros(freq1[np.logical_and(freq1 >= 22050, freq1 <= 24000)].shape), 
                    color='orange', alpha=0.3)


    plt.fill_between(freq1[np.logical_and(freq1 >= 22050, freq1 <= 24000)], 
                    liminf1[np.logical_and(freq1 >= 22050, freq1 <= 24000)]-0.3,
                    -np.ones(freq1[np.logical_and(freq1 >= 22050, freq1 <= 24000)].shape)*160, 
                    color='orange', alpha=0.3)


    #plt.fill_between([22050,24000], [-160,-160],[0,0], color='green', alpha=0.2)

    plt.grid(True)
    plt.ylim((-150,0))
    plt.xlim((0,24300))

    plt.annotate("Threshold \n {ADI, AEI}",
                xy=(5750,-50),
                xytext=(4700,-80 ),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="black",linewidth=2),fontsize=20)


    plt.text(13150,-132,"Different frequency response", fontsize=20)
    plt.annotate("",
                xy=(22050,-132),
                xytext=(19000,-132 ),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="black",linewidth=2))

    plt.annotate("",
                xy=(10000,-132),
                xytext=(13050,-132 ),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="black",linewidth=2))



    plt.text(22300,-130,"$\Delta f_{max}$", fontsize=22)
    plt.annotate("",
                xy=(24000,-132),
                xytext=(23500,-132 ),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="black",linewidth=2))

    plt.annotate("",
                xy=(22050,-132),
                xytext=(22550,-132 ),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="black",linewidth=2))
    fig.axes[0].spines[:].set_color("black")
    

    plt.show()
    
    return fig



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
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
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