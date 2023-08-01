from matplotlib import rcParams
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import datetime
from utils_psql import ConnectDB
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/david/Documentos/Codes/postgresql/")


class PowerPlot():
    def __init__(self):
        self.df = None
        self.cmap = ["#6a3d9a", "#fb9a99", "#33a02c",
                     "#1f78b4", "#FDB462", "#FFFFB3"]

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

    def getData(self):

        path_env = '/home/david/Documentos/Codes/postgresql/.env'
        db = ConnectDB(path_env)

        query = """SELECT audios.folder, audios.name, power.power, power.power_rms
                    FROM audios JOIN power ON audios.cod_audio = power.cod_audio
                    WHERE audios.folder IN ('G0004', 'G0005', 'G0006', 'G0009','G0010','G0015','G0016','G0017', 'G0018','G0019','G0032','G0033','G0034','G0035','G0036')
                    """

        db.execute(query)
        self.df = db.fetchall()

    def plotFigure(self):
        # Crear una figura con subplots
        self.getData()

        self.df['power'] = [10*np.log10(x) for x in self.df['power']]
        self.df['power_rms'] = [10*np.log10(x) for x in self.df['power_rms']]

        fig, ax = plt.subplots(2, 1, figsize=(9, 5))

        self.df.boxplot(column='power', by='folder', ax=ax[0], color='black')
        self.df.boxplot(column='power_rms', by='folder',
                        ax=ax[1], color='black')

        rec = [f'R{n}' for n in range(1, 16)]
        ax[0].set_ylim(-10, -25)
        ax[0].set_xticklabels(rec, rotation=0)
        ax[0].set_xlabel('Recorders')
        ax[0].set_ylabel('Power [dB]')
        ax[0].set_title('')
        fig.suptitle('')

        l1 = ax[0].fill_between([0.5, 5.5], [-10, -10],
                                [-25, -25], alpha=0.2, color=self.cmap[0])
        l2 = ax[0].fill_between([5.5, 10.5], [-10, -10],
                                [-25, -25], alpha=0.2, color=self.cmap[1])
        l3 = ax[0].fill_between([10.5, 15.5], [-10, -10],
                                [-25, -25], alpha=0.2, color=self.cmap[2])

        ax[0].grid(False)

        ax[1].set_ylim(-10, 5)
        ax[1].set_xticklabels(rec, rotation=0)
        ax[1].set_xlabel('Recorders')
        ax[1].set_ylabel('Power [dB]')
        ax[1].set_title('')

        l1 = ax[1].fill_between(
            [0.5, 5.5], [5, 5], [-10, -10], alpha=0.2, color=self.cmap[0])
        l2 = ax[1].fill_between([5.5, 10.5], [5, 5],
                                [-10, -10], alpha=0.2, color=self.cmap[1])
        l3 = ax[1].fill_between([10.5, 15.5], [5, 5],
                                [-10, -10], alpha=0.2, color=self.cmap[2])

        ax[1].grid(False)

        labels = ['AudioMoth v1.0', 'AudioMoth v1.1', 'AudioMoth v1.2']
        fig.legend(handles=[l1, l2, l3], labels=labels, loc='lower center', bbox_to_anchor=(
            0.55, 0.97), ncol=len(labels), fontsize=15)
        fig.savefig(
            '/home/david/Códigos/codes-indices-recorders/results/figures/powerplot.pdf', bbox_inches='tight')

        fig.show()
