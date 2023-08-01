import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/david/Documentos/Codes/postgresql/")
from utils_psql import ConnectDB
import datetime
import matplotlib.ticker as ticker

from matplotlib import rcParams



class TemporalIndicesFigure():
    def __init__(self):
        self.df1 = None
        self.df2 = None
        self.cmap = ["#6a3d9a", "#fb9a99", "#33a02c", "#1f78b4","#FDB462", "#FFFFB3"]

        rcParams["text.usetex"] = True
        rcParams["xtick.labelsize"] = 16
        rcParams["ytick.labelsize"] = 16
        rcParams["figure.constrained_layout.use"] = True
        rcParams["axes.titlesize"] = 20
        rcParams["axes.facecolor"] = "white"
        rcParams["axes.edgecolor"] = "black"
        rcParams["axes.labelsize"] = 18
        rcParams["legend.facecolor"] = "white"
        rcParams["legend.edgecolor"] = "black"
        rcParams['font.family'] = 'serif'

    def format_ticks(self, ax, threshold=1000):
        """
        Formatea los ticks de una gráfica para mostrar números con una "k" si superan el límite especificado.
        
        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): Eje de la gráfica.
            threshold (int): Límite para mostrar los ticks con una "k". Por defecto es 1000.
        """
        def formatter(x, pos):
            if x >= threshold:
                return f'{np.round(x/1000,0)}k'
            else:
                return np.round(x,1)
        for axis in [ax.yaxis]:
            axis.set_major_formatter(ticker.FuncFormatter(formatter))

    def getData(self):

        path_env = '/home/david/Documentos/Codes/postgresql/.env'
        db = ConnectDB(path_env)

        query = """SELECT {t1}.folder,
                    DATE_TRUNC('hour', time) AS hour,
                    AVG(acift) AS acift_avg,STDDEV(acift) AS acift_std,
                    AVG(adi) AS adi_avg, STDDEV(adi) AS adi_std,
                    AVG(beta) AS beta_avg, STDDEV(beta) AS beta_std,
                    AVG(m) AS m_avg, STDDEV(m) AS m_std,
                    AVG(np) AS np_avg, STDDEV(np) AS np_std,
                    AVG(h) AS h_avg, STDDEV(h) AS h_std,
                    AVG(aei) AS aei_avg, STDDEV(aei) AS aei_std,
                    AVG(ndsi) AS ndsi_avg, STDDEV(ndsi) AS ndsi_std
                    FROM {t1} JOIN {t2} ON {t1}.cod_audio = {t2}.cod_audio
                    WHERE {t1}.folder IN ('G60A', 'G60S', 'G21A', 'G21S')
                    GROUP BY folder, hour
                    ORDER BY folder, hour"""
                
        db.execute(query.format(t1='audios', t2='raw_indices'))
        self.df1 = db.fetchall()
        self.df1['hour'] = [x[1].hour.seconds//3600 for x in self.df1.iterrows()]

        db.execute(query.format(t1='audios', t2='processed_indices'))
        self.df2 = db.fetchall()
        self.df2['hour'] = [x[1].hour.seconds//3600 for x in self.df2.iterrows()]



    def plotFigureLinear(self,columns):
        # Crear una figura con subplots
        self.getData()


        fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(8, 10))
        axes = axes.ravel()
        axes = np.reshape(axes,(len(columns),2))

        # Iterar a través de las columnas del DataFrame
        for i, (col, ax) in enumerate(zip(columns,axes )):
            col = f'{col}_avg'
            # Calcular medias y desviaciones estándar por hora del día y carpeta
            means_d1 = self.df1.groupby(['folder', 'hour'])[col].mean().unstack(level=0)
            stds_d1 = self.df1.groupby(['folder', 'hour'])[self.df1.columns[self.df1.columns.get_loc(col) + 1]].mean().unstack(level=0)

            # Calcular medias y desviaciones estándar por hora del día y carpeta
            means_d2 = self.df2.groupby(['folder', 'hour'])[col].mean().unstack(level=0)
            stds_d2 = self.df2.groupby(['folder', 'hour'])[self.df2.columns[self.df2.columns.get_loc(col) + 1]].mean().unstack(level=0)



            lines = []
            labels = []
            # Graficar líneas para cada carpeta y rellenar sombras con desviaciones estándar
            for j, folder in enumerate(self.df1['folder'].unique()):
                ax[0].plot(means_d1.index, means_d1[folder], color= self.cmap[j])
                ax[0].fill_between(means_d1.index, means_d1[folder]-stds_d1[folder], means_d1[folder]+stds_d1[folder], alpha=0.2, color= self.cmap[j])
                

            for j, folder in enumerate(self.df2['folder'].unique()):
                line, = ax[1].plot(means_d2.index, means_d2[folder],color= self.cmap[j])
                ax[1].fill_between(means_d2.index, means_d2[folder]-stds_d2[folder], means_d2[folder]+stds_d2[folder], alpha=0.2, color= self.cmap[j], label=folder)
                if i == len(columns)-1:
                    lines.append(line)
                    if folder == 'G21A' or folder =='G21S':
                        labels.append(f'S2{folder[-1]}')
                    elif folder == 'G60A' or folder == 'G60S' :
                        labels.append(f'S1{folder[-1]}')


            
            self.format_ticks(ax[0])
            self.format_ticks(ax[1])

            # Configurar título y ejes de la figura
            ax[0].set_title(f'Raw {col[:-4].upper()}')
            ax[0].set_xlabel('Hours')
            ax[0].set_ylabel('Value')
            ax[0].set_ylim(bottom=0)

            ax[1].set_title(f'Processed {col[:-4].upper()}')
            ax[1].set_xlabel('Hours')
            ax[1].set_ylabel('Value')
            ax[1].set_ylim(bottom=0)

            if col == 'acift_avg':
                ax[0].set_ylim(650000,760000)
                ax[1].set_ylim(650000,760000)
                ax[0].set_title(f'Raw {col[:-6].upper()}ft')
                ax[1].set_title(f'Processed {col[:-6].upper()}ft')

            if col == 'beta_avg':
                ax[0].set_title(f'Raw $\\beta$')
                ax[1].set_title(f'Processed $\\beta$')

            if i != len(columns)-1:
                ax[0].set_xlabel('')
                ax[1].set_xlabel('')

        # Agregar leyenda
        fig.legend(handles=lines,labels=labels, loc='lower center',bbox_to_anchor=(0.52, -0.05), ncol=len(labels), fontsize=16)

        # Ajustar espacio entre subplots y guardar figura
        fig.tight_layout()
        fig.savefig('/home/david/Documentos/Codes/codes-indices-recorders/results/figures/compare_locations_temp2.pdf',bbox_inches='tight')
        fig.show()
        

    def plotFigurePolar(self,columns):
        # Crear una figura con subplots
        self.getData()


        fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(8, 10),subplot_kw={'projection': 'polar'})
        axes = axes.ravel()
        axes = np.reshape(axes,(len(columns),2))

        # Iterar a través de las columnas del DataFrame
        for i, (col, ax) in enumerate(zip(columns,axes )):
            col = f'{col}_avg'
            # Calcular medias y desviaciones estándar por hora del día y carpeta
            means_d1 = self.df1.groupby(['folder', 'hour'])[col].mean().unstack(level=0)
            stds_d1 = self.df1.groupby(['folder', 'hour'])[self.df1.columns[self.df1.columns.get_loc(col) + 1]].mean().unstack(level=0)

            # Calcular medias y desviaciones estándar por hora del día y carpeta
            means_d2 = self.df2.groupby(['folder', 'hour'])[col].mean().unstack(level=0)
            stds_d2 = self.df2.groupby(['folder', 'hour'])[self.df2.columns[self.df2.columns.get_loc(col) + 1]].mean().unstack(level=0)



            lines = []
            labels = []
            # Graficar líneas para cada carpeta y rellenar sombras con desviaciones estándar
            for j, folder in enumerate(self.df1['folder'].unique()):
                xpolar = 2*np.pi*means_d1.index/means_d1.index.max()
                ax[0].plot(xpolar, means_d1[folder], color= self.cmap[j])
                #ax[0].fill_between(means_d1.index, means_d1[folder]-stds_d1[folder], means_d1[folder]+stds_d1[folder], alpha=0.2, color= self.cmap[j])
                

            for j, folder in enumerate(self.df2['folder'].unique()):
                xpolar = 2*np.pi*means_d2.index/means_d1.index.max()
                line, = ax[1].plot(xpolar, means_d2[folder],color= self.cmap[j])
                #ax[1].fill_between(means_d2.index, means_d2[folder]-stds_d2[folder], means_d2[folder]+stds_d2[folder], alpha=0.2, color= self.cmap[j], label=folder)
                if i == len(columns)-1:
                    lines.append(line)
                    if folder == 'G21A' or folder =='G21S':
                        labels.append(f'S2{folder[-1]}')
                    elif folder == 'G60A' or folder == 'G60S' :
                        labels.append(f'S1{folder[-1]}')


            
            ax[0].set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
            ax[0].set_xticklabels(range(24))  
            ax[0].grid(True)
            # Configurar título y ejes de la figura
            ax[0].set_title(f'Raw {col[:-4].upper()}')
            #ax[0].set_xlabel('Hours')
            #ax[0].set_ylabel('Value')
            ax[0].set_ylim(bottom=0)

            ax[1].set_title(f'Processed {col[:-4].upper()}')
            #ax[1].set_xlabel('Hours')
            #ax[1].set_ylabel('Value')
            ax[1].set_ylim(bottom=0)

            if col == 'acift_avg':
                ax[0].set_ylim(650000,760000)
                ax[1].set_ylim(650000,760000)
                ax[0].set_title(f'Raw {col[:-6].upper()}ft')
                ax[1].set_title(f'Processed {col[:-6].upper()}ft')

            if col == 'beta_avg':
                ax[0].set_title(f'Raw $\\beta$')
                ax[1].set_title(f'Processed $\\beta$')

            if i != len(columns)-1:
                ax[0].set_xlabel('')
                ax[1].set_xlabel('')

        # Agregar leyenda
        fig.legend(handles=lines,labels=labels, loc='lower center',bbox_to_anchor=(0.52, -0.05), ncol=len(labels), fontsize=16)

        # Ajustar espacio entre subplots y guardar figura
        fig.tight_layout()
        fig.show()