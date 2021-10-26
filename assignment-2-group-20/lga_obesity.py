import pandas as pd
import argparse

#Read in dataframe from csv file
obesity_df = pd.read_csv('lga_obesity.csv', encoding='ISO-8859-1')

#Remove rows with municipalities not in metropolitan Melbourne
for index, row in obesity_df.iterrows():
    if ((obesity_df.loc[index, ' lga_name'] != 'Banyule (C)') and (obesity_df.loc[index, ' lga_name'] != 'Bayside (C)') and (obesity_df.loc[index, ' lga_name'] != 'Boroondara (C)') and (obesity_df.loc[index, ' lga_name'] != 'Brimbank (C)') and (obesity_df.loc[index, ' lga_name'] != 'Cardinia (S)') and (obesity_df.loc[index, ' lga_name'] != 'Casey (C)') and (obesity_df.loc[index, ' lga_name'] != 'Greater Dandenong (C)') and (obesity_df.loc[index, ' lga_name'] != 'Darebin (C)') and (obesity_df.loc[index, ' lga_name'] != 'Frankston (C)') and (obesity_df.loc[index, ' lga_name'] != 'Glen Eira (C)') and (obesity_df.loc[index, ' lga_name'] != 'Hobsons Bay (C)') and (obesity_df.loc[index, ' lga_name'] != 'Hume (C)') and (obesity_df.loc[index, ' lga_name'] != 'Kingston (C)') and (obesity_df.loc[index, ' lga_name'] != 'Knox (C)') and (obesity_df.loc[index, ' lga_name'] != 'Manningham (C)') and (obesity_df.loc[index, ' lga_name'] != 'Maribyrnong (C)') and (obesity_df.loc[index, ' lga_name'] != 'Maroondah (C)') and (obesity_df.loc[index, ' lga_name'] != 'Melbourne (C)') and (obesity_df.loc[index, ' lga_name'] != 'Melton (C)') and (obesity_df.loc[index, ' lga_name'] != 'Monash (C)') and (obesity_df.loc[index, ' lga_name'] != 'Moonee Valley (C)') and (obesity_df.loc[index, ' lga_name'] != 'Moreland (C)') and (obesity_df.loc[index, ' lga_name'] != 'Mornington Peninsula (S)') and (obesity_df.loc[index, ' lga_name'] != 'Nillumbik (S)') and (obesity_df.loc[index, ' lga_name'] != 'Port Phillip (C)') and (obesity_df.loc[index, ' lga_name'] != 'Stonnington (C)') and (obesity_df.loc[index, ' lga_name'] != 'Whitehorse (C)') and (obesity_df.loc[index, ' lga_name'] != 'Whittlesea (C)') and (obesity_df.loc[index, ' lga_name'] != 'Wyndham (C)') and (obesity_df.loc[index, ' lga_name'] != 'Yarra (C)') and (obesity_df.loc[index, ' lga_name'] != 'Yarra Ranges (S)')):
        obesity_df.drop(index, inplace=True)

#Create new dataframe with lga name, obesity percentage, area of lga and distance to Melbourne CBD
new_obesity_df = obesity_df[[' lga_name', ' ppl_reporting_being_obese_perc', ' area_of_lga_km2', ' distance_to_melbourne_km']]

#Rename dataframe columns for clarity
new_obesity_df.columns = ['lga_name', 'percentage_obese', 'lga_area_km2', 'distance_to_melbourne_km']

#Create csv file for dataframe
new_obesity_df.to_csv('obesity.csv', index=False)