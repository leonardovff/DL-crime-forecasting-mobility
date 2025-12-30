import pandas as pd
import numpy as np
import os

"""## Select only the 6 crime types and make naming compatible for all 5 cities

### 1. Arapiraca
"""

city_folder = 'Arapiraca'

# load dataset clean
df_clean = pd.read_csv(f'./preprocessing/crime_data/Preprocessing_raw_data_per_city/{city_folder}/{city_folder}_crimes_clean_2012_to_2020.csv', low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

list_crimes = [
    'ROUBO DE VEÍCULO (MOTO)', # x
    'ROUBO A CASA COMERCIAL', # x
    'ROUBO A TRANSEUNTE', # x
    'ROUBO OUTROS', # x
    'ROUBO A RESIDÊNCIA', # x
    'ROUBO DE VEÍCULO (DE PASSEIO)', # x
    'ROUBO DE VEÍCULO (OUTROS)', # x
    # 'ROUBO DE CARGAS',
    'TENTATIVA DE ROUBO', # x
    # 'ROUBO A TRANSPORTE COLETIVO RODOVIÁRIO', 
    # 'ROUBO A TRANSPORTE COLETIVO URBANO', 
    'EXTORSÃO', # x
    # 'EXTORSÃO MEDIANTE SEQUESTRO', 
    # 'ROUBO A CORRESPONDENTE BANCÁRIO', 
    #'ARROMBAMENTO A CAIXA ELETRÔNICO (MAÇARICO)',
    'AGG. ASSAULT','COMMON ASSAULT','AUTO THEFT','BURGLARY','HOMICIDE','ROBBERY'
]

# keep only rows with those 6 crimes
df_filtered = df_clean.copy()
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
df_filtered['crime_type'].replace("ROUBO DE VEÍCULO (MOTO)", "Motor Vehicle Theft", inplace=True)
df_filtered['crime_type'].replace("ROUBO DE VEÍCULO (DE PASSEIO)", "Motor Vehicle Theft", inplace=True)
df_filtered['crime_type'].replace("ROUBO DE VEÍCULO (OUTROS)", "Motor Vehicle Theft", inplace=True)
df_filtered['crime_type'].replace("ROUBO DE VEÍCULO (OUTROS)", "Motor Vehicle Theft", inplace=True)
df_filtered['crime_type'].replace("ROUBO A TRANSEUNTE", "Robbery", inplace=True)
df_filtered['crime_type'].replace("ROUBO OUTROS", "Robbery",inplace=True)
df_filtered['crime_type'].replace("TENTATIVA DE ROUBO", "Robbery", inplace=True)
df_filtered['crime_type'].replace("EXTORSÃO", "Robbery", inplace=True)
df_filtered['crime_type'].replace("ROUBO A CASA COMERCIAL", "Burglary", inplace=True)
df_filtered['crime_type'].replace("ROUBO A RESIDÊNCIA", "Burglary", inplace=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.title()


# save final dataset
os.makedirs('Crime_data_outputs/', exist_ok=True)
df_filtered.to_csv(f'Crime_data_outputs/{city_folder}_selected_crimes_clean_all.csv')
print("Saved dataset")
print(df_filtered['crime_type'].unique().tolist())
