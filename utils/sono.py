import pandas as pd


def merge_df(df1, labels):
    df = pd.DataFrame()
    df['Grabaciones'] = df1.groupby('File').count()['Class'].index
    df['value'] = df1.groupby('File').count()['Class'].values

    #merge Gn1 with labels on  Grabaciones column
    df = pd.merge(df, labels, on='Grabaciones')
    df = df[df['Labels'] == 0]
    return df

def get_datetime(df,col,prefix,extension):
    df['datetime'] = df[col].str.replace(prefix, '')
    df['datetime'] = df['datetime'].str.replace(extension, '')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d_%H%M%S')
    return df

def get_dataplot(df,col1,col2,group = 'hour'):
    if group == 'hour':
        datay = df.groupby(df[col1].dt.hour).mean()[col2].values
        datax = df.groupby(df[col1].dt.hour).mean()[col2].index
    elif group == 'day':
        datay = df.groupby(df[col1].dt.day).mean()[col2].values
        datax = df.groupby(df[col1].dt.day).mean()[col2].index
    elif group == 'month':
        datay = df.groupby(df[col1].dt.month).mean()[col2].values
        datax = df.groupby(df[col1].dt.month).mean()[col2].index
    return datax, datay

def load_index(path, name):
    file = path.joinpath(name)
    df = pd.read_excel(file)
    df = df.drop(columns=['Unnamed: 0'])
    return df

def graph_index(path, names, index, group = 'hour'):
    xt = []
    yt = []
    for n in names:
        df = load_index(path, n)
        prefix = f'{df.iloc[0]["name"].split("_")[0]}_'
        ext = f'{df.iloc[0]["name"][-4:]}'
        dt = get_datetime(df,'name',prefix,ext)
        x,y = get_dataplot(dt,'datetime',index,group)
        xt.append(x)
        yt.append(y)
    return xt, yt