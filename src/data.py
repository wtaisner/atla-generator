import pandas as pd


def read_dataframe(path: str = '../data/avatar.csv') -> pd.DataFrame:
    """
    reads data and performs basic preprocessing
    :param path: pth to the avatar.csv file
    :return: preprocessed dataframe
    """
    df = pd.read_csv(path, encoding='unicode_escape').drop(columns=['Unnamed: 0', 'id'])
    df = df[df['character'] != 'Scene Description']
    df = df[~df['character'].str.contains('and')]
    df['character'] = df['character'].str.lower()
    df['character'] = df['character'].str.replace(':|actor|actress', '', regex=True)
    df['character'] = df['character'].str.title()
    df = df.reset_index()
    df = df.drop(columns=['index'])
    return df
