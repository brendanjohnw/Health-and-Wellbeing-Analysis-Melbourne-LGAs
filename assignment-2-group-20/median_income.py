import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split

#Read in dataframe from csv file
income_df = pd.read_csv('lga_income.csv', encoding='ISO-8859-1')

#Remove rows with municipalities not in metropolitan Melbourne
for index, row in income_df.iterrows():
    if ((income_df.loc[index, ' lga_name16'] != 'Banyule (C)') and (income_df.loc[index, ' lga_name16'] != 'Bayside (C)') and (income_df.loc[index, ' lga_name16'] != 'Boroondara (C)') and (income_df.loc[index, ' lga_name16'] != 'Brimbank (C)') and (income_df.loc[index, ' lga_name16'] != 'Cardinia (S)') and (income_df.loc[index, ' lga_name16'] != 'Casey (C)') and (income_df.loc[index, ' lga_name16'] != 'Greater Dandenong (C)') and (income_df.loc[index, ' lga_name16'] != 'Darebin (C)') and (income_df.loc[index, ' lga_name16'] != 'Frankston (C)') and (income_df.loc[index, ' lga_name16'] != 'Glen Eira (C)') and (income_df.loc[index, ' lga_name16'] != 'Hobsons Bay (C)') and (income_df.loc[index, ' lga_name16'] != 'Hume (C)') and (income_df.loc[index, ' lga_name16'] != 'Kingston (C) (Vic.)') and (income_df.loc[index, ' lga_name16'] != 'Knox (C)') and (income_df.loc[index, ' lga_name16'] != 'Manningham (C)') and (income_df.loc[index, ' lga_name16'] != 'Maribyrnong (C)') and (income_df.loc[index, ' lga_name16'] != 'Maroondah (C)') and (income_df.loc[index, ' lga_name16'] != 'Melbourne (C)') and (income_df.loc[index, ' lga_name16'] != 'Melton (C)') and (income_df.loc[index, ' lga_name16'] != 'Monash (C)') and (income_df.loc[index, ' lga_name16'] != 'Moonee Valley (C)') and (income_df.loc[index, ' lga_name16'] != 'Moreland (C)') and (income_df.loc[index, ' lga_name16'] != 'Mornington Peninsula (S)') and (income_df.loc[index, ' lga_name16'] != 'Nillumbik (S)') and (income_df.loc[index, ' lga_name16'] != 'Port Phillip (C)') and (income_df.loc[index, ' lga_name16'] != 'Stonnington (C)') and (income_df.loc[index, ' lga_name16'] != 'Whitehorse (C)') and (income_df.loc[index, ' lga_name16'] != 'Whittlesea (C)') and (income_df.loc[index, ' lga_name16'] != 'Wyndham (C)') and (income_df.loc[index, ' lga_name16'] != 'Yarra (C)') and (income_df.loc[index, ' lga_name16'] != 'Yarra Ranges (S)')):
        income_df.drop(index, inplace=True)
        
#Create new dataframe with lga name and median income 
new_income_df = income_df[[' lga_name16', ' median_aud']]

#Rename dataframe columns for clarity
new_income_df.columns = ['lga_name', 'median_income_aud']

#Remove brackets and whitespace from lga names
def remove_brackets(lga_name):
    lga_name = (re.sub(r' \([^)]*\)', '', lga_name))
    return lga_name

#Merge renamed lga dataframe with income statistics
income_stats_df = new_income_df[['median_income_aud']]
renamed_lga_df = new_income_df['lga_name'].apply(remove_brackets)
modified_income_df = pd.concat([renamed_lga_df, income_stats_df], join='outer', axis=1)

#Read in obesity statistics dataframe from csv file
obesity_merged_stats_df = pd.read_csv('obesity_merged_stats.csv', encoding='ISO-8859-1')

#Merge renamed lga dataframe with obesity statistics
obesity_stats_df = obesity_merged_stats_df[['percentage_obese', 'population_2015', 'lga_area_km2', 'distance_to_melbourne_km']]
renamed_obesity_lga_df = obesity_merged_stats_df['lga_name'].apply(remove_brackets)
modified_obesity_df =  pd.concat([renamed_obesity_lga_df, obesity_stats_df], join='outer', axis=1)

#Merge modified obesity dataframe with modified income dataframe
final_income_df = pd.merge(left=modified_income_df, right=modified_obesity_df, left_on='lga_name', right_on='lga_name')

#Sort dataframe based on distance to CBD in descending order
sorted_income_df = final_income_df.sort_values(['distance_to_melbourne_km'], ascending=True)

#Colour code each row based on distance to Melbourne CBD
def colour_code(row):
    if (0 <= row['distance_to_melbourne_km'] < 10):
        colour = 'green'
    elif (10 <= row['distance_to_melbourne_km'] < 20):
        colour = 'yellow'
    elif (20 <= row['distance_to_melbourne_km'] < 30):
        colour = 'orange'
    elif (30 <= row['distance_to_melbourne_km'] < 40):
        colour = 'red'
    elif (40 <= row['distance_to_melbourne_km'] < 50):
        colour = 'purple'
    elif (50 <= row['distance_to_melbourne_km'] < 60):
        colour = 'black'
    return colour

#Colour code each lga based on distance to Melbourne CBD
sorted_income_df['colour'] = sorted_income_df.apply(colour_code, axis=1)
sorted_income_df.reset_index(drop=True, inplace=True)

#Create a list containing labels for distance range to Melbourne
distance = ['Within 10km to CBD', 'Within 20km to CBD', 'Within 30km to CBD', 'Within 40km to CBD', 'Within 50km to CBD', 'Within 60km to CBD']
distance_index = 0

#Iterate through each row of dataframe and plot median income vs percentage obese, colour coding based on lga distance to Melbourne
for index, row in sorted_income_df.iterrows():
    if index == 0:
        plt.scatter(sorted_income_df.iloc[index, 1], sorted_income_df.iloc[index, 2], color=sorted_income_df.iloc[index, 6], label=distance[distance_index])
    elif (sorted_income_df.iloc[index, 6] == sorted_income_df.iloc[index-1, 6]):
        plt.scatter(sorted_income_df.iloc[index, 1], sorted_income_df.iloc[index, 2], color=sorted_income_df.iloc[index, 6])
    else:
        distance_index += 1
        plt.scatter(sorted_income_df.iloc[index, 1], sorted_income_df.iloc[index, 2], color=sorted_income_df.iloc[index, 6], label=distance[distance_index])
#Annotate each scatter point with respective lga name
for i, txt in enumerate(sorted_income_df['lga_name']):
    text = plt.annotate(txt, (sorted_income_df['median_income_aud'][i], sorted_income_df['percentage_obese'][i]))
    text.set_fontsize(11)
#Resize scatter plot
plt.gcf().set_size_inches((17, 17)) 
#Create a legend for the scatter plot
plt.legend(prop={'size': 16})
#Limit x and y axis
plt.xlim(34250, 54250)
plt.ylim(7.5, 27.5)
#Label the x and y-axis for the scatter plot
plt.xlabel('Median Income (AUD)')
plt.ylabel('Population Obese (%)')
#Create a title for the scatter plot
plt.title('Median Income Verses Obesity')
plt.savefig('median_income_vs_obesity_scatter.png', bbox_inches='tight')
plt.show()

#Clusters data points in scatter plot
def clustering(df,k):
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
    kmeans = KMeans(n_clusters = k)
    points = df.values
    #Fitting the Kmeans object to the DataFrame
    kmeans.fit(points)
    #Initializing the centroids
    clusters = kmeans.cluster_centers_
    y_kmeans = kmeans.fit_predict(points) #Value assigned to each data point
    plt.scatter(points[y_kmeans == 0,0], points[y_kmeans == 0,1], s=50, color = 'red')
    plt.scatter(points[y_kmeans == 1,0], points[y_kmeans == 1,1], s=50, color = 'blue')
    plt.scatter(points[y_kmeans == 2,0], points[y_kmeans == 2,1], s=50, color = 'green')
    plt.scatter(points[y_kmeans == 3,0], points[y_kmeans == 3,1], s=50, color = 'yellow')
    plt.scatter(points[y_kmeans == 4,0], points[y_kmeans == 4,1], s=50, color = 'black')
    
#Cluster data points in scatter plot for median income (AUD) vs percentage obese
clustering(sorted_income_df[['median_income_aud', 'percentage_obese']], 4)
#Annotate each scatter point with respective lga name
for i, txt in enumerate(sorted_income_df['lga_name']):
    text = plt.annotate(txt, (sorted_income_df['median_income_aud'][i], sorted_income_df['percentage_obese'][i]))
    text.set_fontsize(11)
#Resize scatter plot
plt.gcf().set_size_inches((17, 17))
#Limit x and y axis
plt.xlim(34250, 54250)
plt.ylim(7.5, 27.5)
#Label the x and y-axis for the scatter plot
plt.xlabel('Median Income (AUD)')
plt.ylabel('Population Obese (%)')
#Create a title for the scatter plot
plt.title('Median Income Verses Obesity')
plt.savefig('median_income_vs_obesity_cluster.png', bbox_inches='tight')
plt.show()

#Classify obesity percentage into intervals
def obesity_classification(obesity_df):
    if (7.5 <= obesity_df['percentage_obese'] < 12.5):
        risk = 1
    elif (12.5 <= obesity_df['percentage_obese'] < 17.5):
        risk = 2
    elif (17.5 <= obesity_df['percentage_obese'] < 22.5):
        risk = 3
    elif (22.5 <= obesity_df['percentage_obese'] <= 27.5):
        risk = 4
    return int(risk)

#Classify percentage of population with inadequate fruit and vegetable consumption into intervals
def median_income_classification(median_income_df):
    if (35052 <= median_income_df['median_income_aud'] < 39501.75):
        risk = 1
    elif (39501.75 <= median_income_df['median_income_aud'] < 43951.5):
        risk = 2
    elif (43951.5 <= median_income_df['median_income_aud'] < 48401.25):
        risk = 3
    elif (48401.25 <= median_income_df['median_income_aud'] <= 52851):
        risk = 4
    return int(risk)

#Calculate mutual information score
sorted_income_df['independent_class'] = sorted_income_df.apply(median_income_classification, axis=1)
sorted_income_df['dependent_class'] = sorted_income_df.apply(obesity_classification, axis=1)
df = sorted_income_df[['independent_class', 'dependent_class']]
independent = df['independent_class']
dependent = df['dependent_class']
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, train_size=0.80, test_size=0.20, random_state=42)
#Information gain
mutual_info = mutual_info_score(X_train, y_train)
print(f"Mutual Information Score: {round(mutual_info, 2)}")