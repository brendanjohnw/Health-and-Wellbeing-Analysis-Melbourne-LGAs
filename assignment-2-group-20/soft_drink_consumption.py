import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Read in dataframe from csv file
soft_drink_df = pd.read_csv('lga_soft_drink_consumption.csv', encoding='ISO-8859-1')

#Remove rows with municipalities not in metropolitan Melbourne
for index, row in soft_drink_df.iterrows():
    if ((soft_drink_df.loc[index, ' lga_name06'] != 'Banyule (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Bayside (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Boroondara (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Brimbank (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Cardinia (S)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Casey (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Greater Dandenong (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Darebin (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Frankston (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Glen Eira (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Hobsons Bay (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Hume (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Kingston (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Knox (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Manningham (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Maribyrnong (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Maroondah (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Melbourne (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Melton (S)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Monash (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Moonee Valley (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Moreland (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Mornington Peninsula (S)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Nillumbik (S)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Port Phillip (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Stonnington (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Whitehorse (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Whittlesea (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Wyndham (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Yarra (C)') and (soft_drink_df.loc[index, ' lga_name06'] != 'Yarra Ranges (S)')):
        soft_drink_df.drop(index, inplace=True)
        
#Create new dataframe with lga name, percentage of people consuming soft drinks and victoria average percentage consumption
new_soft_drink_df = soft_drink_df[[' lga_name06', ' numeric', ' vic_ave']]

#Rename dataframe columns for clarity
new_soft_drink_df.columns = ['lga_name', 'percentage_consume_soft_drink', 'vic_avg_perc_consume_soft_drink']

#Remove brackets and whitespace from lga names
def remove_brackets(lga_name):
    lga_name = (re.sub(r' \([^)]*\)', '', lga_name))
    return lga_name

#Merge renamed lga dataframe with soft drink statistics
soft_drink_stats_df = new_soft_drink_df[['percentage_consume_soft_drink', 'vic_avg_perc_consume_soft_drink']]
renamed_lga_df = new_soft_drink_df['lga_name'].apply(remove_brackets)
modified_soft_drink_df = pd.concat([renamed_lga_df, soft_drink_stats_df], join='outer', axis=1)

#Read in obesity statistics dataframe from csv file
obesity_merged_stats_df = pd.read_csv('obesity_merged_stats.csv', encoding='ISO-8859-1')

#Merge renamed lga dataframe with obesity statistics
obesity_stats_df = obesity_merged_stats_df[['percentage_obese', 'population_2015', 'lga_area_km2', 'distance_to_melbourne_km']]
renamed_obesity_lga_df = obesity_merged_stats_df['lga_name'].apply(remove_brackets)
modified_obesity_df =  pd.concat([renamed_obesity_lga_df, obesity_stats_df], join='outer', axis=1)

#Merge modified obesity dataframe with modified soft drink dataframe
final_soft_drink_df = pd.merge(left=modified_soft_drink_df, right=modified_obesity_df, left_on='lga_name', right_on='lga_name')

#Sort dataframe based on distance to CBD in descending order
sorted_soft_drink_df = final_soft_drink_df.sort_values(['distance_to_melbourne_km'], ascending=True)

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
sorted_soft_drink_df['colour'] = sorted_soft_drink_df.apply(colour_code, axis=1)
sorted_soft_drink_df.reset_index(drop=True, inplace=True)

#Create a list containing labels for distance range to Melbourne
distance = ['Within 10km to CBD', 'Within 20km to CBD', 'Within 30km to CBD', 'Within 40km to CBD', 'Within 50km to CBD', 'Within 60km to CBD']
distance_index = 0
#Iterate through each row of dataframe and plot percentage of population that consume soft drinks vs percentage obese, colour coding based on lga distance to Melbourne
for index, row in sorted_soft_drink_df.iterrows():
    if index == 0:
        plt.scatter(sorted_soft_drink_df.iloc[index, 1], sorted_soft_drink_df.iloc[index, 3], color=sorted_soft_drink_df.iloc[index, 7], label=distance[distance_index])
    elif (sorted_soft_drink_df.iloc[index, 7] == sorted_soft_drink_df.iloc[index-1, 7]):
        plt.scatter(sorted_soft_drink_df.iloc[index, 1], sorted_soft_drink_df.iloc[index, 3], color=sorted_soft_drink_df.iloc[index, 7])
    else:
        distance_index += 1
        plt.scatter(sorted_soft_drink_df.iloc[index, 1], sorted_soft_drink_df.iloc[index, 3], color=sorted_soft_drink_df.iloc[index, 7], label=distance[distance_index])
#Annotate each scatter point with respective lga name
for i, txt in enumerate(sorted_soft_drink_df['lga_name']):
    text = plt.annotate(txt, (sorted_soft_drink_df['percentage_consume_soft_drink'][i], sorted_soft_drink_df['percentage_obese'][i]))
    text.set_fontsize(11)
#Resize scatter plot
plt.gcf().set_size_inches((17, 17)) 
#Create a legend for the scatter plot
plt.legend(prop={'size': 16})
#Limit x and y axis
plt.xlim(4, 21.5)
plt.ylim(7.5, 27.5)
#Label the x and y-axis for the scatter plot
plt.xlabel('Population That Consume Soft Drinks Frequently (%)')
plt.ylabel('Population Obese (%)')
#Create a title for the scatter plot
plt.title('Soft Drink Consumption Verses Obesity')
plt.savefig('soft_drink_vs_obesity_scatter.png', bbox_inches='tight')
plt.show()

#Apply Pearson's correlation
soft_drink_pearsons = sorted_soft_drink_df[['percentage_consume_soft_drink', 'percentage_obese']]
correlation_score = soft_drink_pearsons.corr(method='pearson', min_periods=1)
print(correlation_score)

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
    plt.scatter(points[y_kmeans == 0,0], points[y_kmeans == 0,1], s=25, color = 'red')
    plt.scatter(points[y_kmeans == 1,0], points[y_kmeans == 1,1], s=25, color = 'blue')
    plt.scatter(points[y_kmeans == 2,0], points[y_kmeans == 2,1], s=25, color = 'green')
    plt.scatter(points[y_kmeans == 3,0], points[y_kmeans == 3,1], s=25, color = 'yellow')
    plt.scatter(points[y_kmeans == 4,0], points[y_kmeans == 4,1], s=25, color = 'black')
    
#Cluster data points in scatter plot for percentage of population that consumes soft drink vs percentage obese
clustering(sorted_soft_drink_df[['percentage_consume_soft_drink', 'percentage_obese']], 4)
#Annotate each scatter point with respective lga name
for i, txt in enumerate(sorted_soft_drink_df['lga_name']):
    text = plt.annotate(txt, (sorted_soft_drink_df['percentage_consume_soft_drink'][i], sorted_soft_drink_df['percentage_obese'][i]))
    text.set_fontsize(11)
#Resize scatter plot
plt.gcf().set_size_inches((17, 17)) 
#Limit x and y axis
plt.xlim(4, 21.5)
plt.ylim(7.5, 27.5)
#Label the x and y-axis for the scatter plot
plt.xlabel('Population That Consume Soft Drinks Frequently (%)')
plt.ylabel('Population Obese (%)')
#Create a title for the scatter plot
plt.title('Soft Drink Consumption Verses Obesity')
plt.savefig('soft_drink_vs_obesity_cluster.png', bbox_inches='tight')
plt.show()

#Plot data points with linear regression line
sorted_soft_drink_df.corr()
independent_var = pd.DataFrame(sorted_soft_drink_df['percentage_consume_soft_drink']) 
dependent_var = pd.DataFrame(sorted_soft_drink_df['percentage_obese'])
#Train test split
X_train,X_test, y_train, y_test= train_test_split(independent_var, dependent_var, train_size=0.80, test_size=0.20, random_state=42)
#Build linear regression model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
model.coef_
model.intercept_
model.score(independent_var, dependent_var) #R^2 value score
sorted_soft_drink_df.plot(kind='scatter', x='percentage_consume_soft_drink', y='percentage_obese')
#Print the regression formula
m = round(float(model.coef_), 2)
c = round(float(model.intercept_), 2)
plt.plot(independent_var, model.predict(independent_var), color = 'red', linewidth = 1, label='y={:.2f}x+{:.2f}'.format(m,c))
print(f"Equation of the regression line: y = {m}x+{c}")
#Limit x and y axis
plt.xlim(3.5, 21.5)
plt.ylim(6.5, 29)
#Label the x and y-axis for the scatter plot
plt.xlabel('Population That Consume Soft Drinks Frequently (%)')
plt.ylabel('Population Obese (%)')
#Create a legend for regression formula
plt.legend(fontsize=9)
#Create a title 
plt.title('Soft Drink Consumption Verses Obesity')
plt.savefig('soft_drink_vs_obesity_linear_regression.png', bbox_inches='tight')
plt.show()