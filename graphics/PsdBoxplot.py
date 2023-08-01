import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/david/Documentos/Codes/postgresql/")
from utils_psql import ConnectDB
import datetime
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib import rcParams



class PsdBoxplot():
    def __init__(self):
        self.df = None
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

    def getData(self):

        path_env = '/home/david/Documentos/Codes/postgresql/.env'
        db = ConnectDB(path_env)

        query = """
                SELECT folder, psd
                FROM audios 
                JOIN raw_psd ON audios.cod_audio = raw_psd.cod_audio
                WHERE folder IN ('G0004', 'G0005', 'G0006', 'G0009','G0010','G0015','G0016','G0017', 'G0018','G0019','G0032','G0033','G0034','G0035','G0036')
                """
                
        db.execute(query)
        self.df = db.fetchall()
        
    def plotFigure(self):
        # Crear una figura con subplots
        self.getData()


        self.df['psd'] = [20*np.log(np.array(x)/np.array(x).max()) for x in self.df['psd']]

        fig, ax = plt.subplots(1,1,figsize=(8,4))


        for i, f in enumerate(sorted(self.df.folder.unique())):
            x = self.df[self.df.folder == f].psd.mean()
            if i < 5:
                color = self.cmap[0]
                l1, = ax.plot(np.linspace(0, 24000,257),x, color=color, alpha=0.6, label = f)
            elif i < 10 and i >= 5:
                color = self.cmap[1]
                l2, = ax.plot(np.linspace(0, 24000,257),x, color=color, alpha=0.6, label = f)
            else:
                color = self.cmap[2]
                l3, = ax.plot(np.linspace(0, 24000,257),x, color=color, alpha=0.6, label = f)

            
        

        labels = ['Audiomoth v1.0', 'Audiomoth v1.1', 'Audiomoth v1.2']
        fig.legend(handles=[l1,l2,l3],labels=labels, loc='lower center',bbox_to_anchor=(0.55, -0.13), ncol=len(labels), fontsize=15)


        ax.grid(True,color='gray', alpha=0.4)
        ax.set_xlim(0,24000)
        ax.set_ylim(-140,0)



        #ax.set_xticklabels(np.linspace(0, 24000,257))


        fig.show()