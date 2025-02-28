import pandas as pd
def getdata():
    data=pd.read_csv('startup.csv')
    print(data.head())
    print("###########################################")
    print(data.info())
    print("###########################################")
    print(data.describe(include='all'))
    print("###########################################")
    print(data.shape)
    print("###########################################")
    print(data.isnull().sum())
    return data
# getdata()
