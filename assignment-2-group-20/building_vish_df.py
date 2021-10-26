import pandas as pd
import argparse

#Read in dataframe from csv file
obesity_df = pd.read_csv('./ABS_-_Regional_Population__LGA__2001-2019.csv', encoding='ISO-8859-1')
old_df = pd.read_csv('./obesity.csv', encoding='ISO-8859-1')

#Remove rows with municipalities not in metropolitan Melbourne
for index, row in obesity_df.iterrows():
    if ((obesity_df.loc[index, ' lga_name'] != 'Banyule (C)') and (obesity_df.loc[index, ' lga_name'] != 'Bayside (C)') and (obesity_df.loc[index, ' lga_name'] != 'Boroondara (C)') and (obesity_df.loc[index, ' lga_name'] != 'Brimbank (C)') and (obesity_df.loc[index, ' lga_name'] != 'Cardinia (S)') and (obesity_df.loc[index, ' lga_name'] != 'Casey (C)') and (obesity_df.loc[index, ' lga_name'] != 'Greater Dandenong (C)') and (obesity_df.loc[index, ' lga_name'] != 'Darebin (C)') and (obesity_df.loc[index, ' lga_name'] != 'Frankston (C)') and (obesity_df.loc[index, ' lga_name'] != 'Glen Eira (C)') and (obesity_df.loc[index, ' lga_name'] != 'Hobsons Bay (C)') and (obesity_df.loc[index, ' lga_name'] != 'Hume (C)') and (obesity_df.loc[index, ' lga_name'] != 'Kingston (C)') and (obesity_df.loc[index, ' lga_name'] != 'Knox (C)') and (obesity_df.loc[index, ' lga_name'] != 'Manningham (C)') and (obesity_df.loc[index, ' lga_name'] != 'Maribyrnong (C)') and (obesity_df.loc[index, ' lga_name'] != 'Maroondah (C)') and (obesity_df.loc[index, ' lga_name'] != 'Melbourne (C)') and (obesity_df.loc[index, ' lga_name'] != 'Melton (C)') and (obesity_df.loc[index, ' lga_name'] != 'Monash (C)') and (obesity_df.loc[index, ' lga_name'] != 'Moonee Valley (C)') and (obesity_df.loc[index, ' lga_name'] != 'Moreland (C)') and (obesity_df.loc[index, ' lga_name'] != 'Mornington Peninsula (S)') and (obesity_df.loc[index, ' lga_name'] != 'Nillumbik (S)') and (obesity_df.loc[index, ' lga_name'] != 'Port Phillip (C)') and (obesity_df.loc[index, ' lga_name'] != 'Stonnington (C)') and (obesity_df.loc[index, ' lga_name'] != 'Whitehorse (C)') and (obesity_df.loc[index, ' lga_name'] != 'Whittlesea (C)') and (obesity_df.loc[index, ' lga_name'] != 'Wyndham (C)') and (obesity_df.loc[index, ' lga_name'] != 'Yarra (C)') and (obesity_df.loc[index, ' lga_name'] != 'Yarra Ranges (S)')):
        obesity_df.drop(index, inplace=True)

# Precess and Clean population dataset
new_obesity_df = obesity_df[['erp_2015',' lga_name']]
sorted_new_obesity_df = new_obesity_df.sort_values( by= ' lga_name')
sorted_new_obesity_df.columns = ['population_2015','lga_name'] 

# Merge onto universal_df
universal_df = pd.merge(old_df, sorted_new_obesity_df, on='lga_name')
universal_df.to_csv('universal_df.csv', index=False)






# Read files
obesity_df = pd.read_csv("./fitness-centres.csv", encoding='ISO-8859-1')
obesity_df = obesity_df[['LGA']]
obesity_df.columns = [' lga_name'] 

old_df = pd.read_csv('./obesity_merged_stats.csv', encoding='ISO-8859-1')



#Remove rows with municipalities not in metropolitan Melbourne
for index, row in obesity_df.iterrows():
    if ((obesity_df.loc[index, ' lga_name'] != 'Banyule (C)') and (obesity_df.loc[index, ' lga_name'] != 'Bayside (C)') and (obesity_df.loc[index, ' lga_name'] != 'Boroondara (C)') and (obesity_df.loc[index, ' lga_name'] != 'Brimbank (C)') and (obesity_df.loc[index, ' lga_name'] != 'Cardinia (S)') and (obesity_df.loc[index, ' lga_name'] != 'Casey (C)') and (obesity_df.loc[index, ' lga_name'] != 'Greater Dandenong (C)') and (obesity_df.loc[index, ' lga_name'] != 'Darebin (C)') and (obesity_df.loc[index, ' lga_name'] != 'Frankston (C)') and (obesity_df.loc[index, ' lga_name'] != 'Glen Eira (C)') and (obesity_df.loc[index, ' lga_name'] != 'Hobsons Bay (C)') and (obesity_df.loc[index, ' lga_name'] != 'Hume (C)') and (obesity_df.loc[index, ' lga_name'] != 'Kingston (C)') and (obesity_df.loc[index, ' lga_name'] != 'Knox (C)') and (obesity_df.loc[index, ' lga_name'] != 'Manningham (C)') and (obesity_df.loc[index, ' lga_name'] != 'Maribyrnong (C)') and (obesity_df.loc[index, ' lga_name'] != 'Maroondah (C)') and (obesity_df.loc[index, ' lga_name'] != 'Melbourne (C)') and (obesity_df.loc[index, ' lga_name'] != 'Melton (C)') and (obesity_df.loc[index, ' lga_name'] != 'Monash (C)') and (obesity_df.loc[index, ' lga_name'] != 'Moonee Valley (C)') and (obesity_df.loc[index, ' lga_name'] != 'Moreland (C)') and (obesity_df.loc[index, ' lga_name'] != 'Mornington Peninsula (S)') and (obesity_df.loc[index, ' lga_name'] != 'Nillumbik (S)') and (obesity_df.loc[index, ' lga_name'] != 'Port Phillip (C)') and (obesity_df.loc[index, ' lga_name'] != 'Stonnington (C)') and (obesity_df.loc[index, ' lga_name'] != 'Whitehorse (C)') and (obesity_df.loc[index, ' lga_name'] != 'Whittlesea (C)') and (obesity_df.loc[index, ' lga_name'] != 'Wyndham (C)') and (obesity_df.loc[index, ' lga_name'] != 'Yarra (C)') and (obesity_df.loc[index, ' lga_name'] != 'Yarra Ranges (S)')):
        obesity_df.drop(index, inplace=True)

obesity_df = obesity_df.groupby(' lga_name').size()
obesity_df = obesity_df.reset_index(level=0)
obesity_df.columns = ['lga_name', 'gym_count']

missing_lga = {'lga_name': "Nillumbik (S)", 'gym_count': 0}
obesity_df = obesity_df.append(missing_lga, ignore_index=True)

# Precess and Clean population dataset
sorted_new_obesity_df = obesity_df.sort_values( by= 'lga_name')

# Merge onto universal_df
universal_df = pd.merge(old_df, sorted_new_obesity_df, on='lga_name')

# Add Gym# / km^2 data
data = universal_df['gym_count'].div(universal_df['lga_area_km2'])
data = data.round(decimals=5)
universal_df = universal_df.assign(gyms_per_km2=data.values)

universal_df






# Read files
obesity_df = universal_df

#Colour code each row based on distance to Melbourne CBD
def distance_bin(row):
    if (0 <= row['distance_to_melbourne_km'] < 10):
        text = 'Within 0-10 km to CBD'
    elif (10 <= row['distance_to_melbourne_km'] < 20):
        text = 'Within 10-20 km to CBD'
    elif (20 <= row['distance_to_melbourne_km'] < 30):
        text = 'Within 20-30 km to CBD'
    elif (30 <= row['distance_to_melbourne_km'] < 40):
        text = 'Within 30-40 km to CBD'
    elif (40 <= row['distance_to_melbourne_km'] < 50):
        text = 'Within 40-50 km to CBD'
    elif (50 <= row['distance_to_melbourne_km'] < 60):
        text = 'Within 50-60 km to CBD'
    return text

#Colour code each lga based on distance to Melbourne CBD
obesity_df['CBD_distance_bin'] = obesity_df.apply(distance_bin, axis=1)
obesity_df.reset_index(drop=True, inplace=True)

obesity_df.to_csv('new2_universal_df.csv', index=False)