# Merge all relevant data sets into one dataframe

import os
import pandas as pd
from matplotlib import pyplot as plt
import re
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split



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

def clustering(df,k):
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
    kmeans = KMeans(n_clusters = k)
    points = df.values
    # fitting the Kmeans object to the DataFrame
    kmeans.fit(points)
    # initializing the centroids
    clusters = kmeans.cluster_centers_
    y_kmeans = kmeans.fit_predict(points) # value assigned to each data point
    plt.scatter(points[y_kmeans == 0,0], points[y_kmeans == 0,1], s=50, color = 'red')
    plt.scatter(points[y_kmeans == 1,0], points[y_kmeans == 1,1], s=50, color = 'blue')
    plt.scatter(points[y_kmeans == 2,0], points[y_kmeans == 2,1], s=50, color = 'green')
    plt.scatter(points[y_kmeans == 3,0], points[y_kmeans == 3,1], s=50, color = 'yellow')
    plt.scatter(points[y_kmeans == 4,0], points[y_kmeans == 4,1], s=50, color = 'black')
    
def remove_brackets(lga_name):
    lga_name = (re.sub(r' \([^)]*\)', '', lga_name))
    return lga_name

def classify_activity(activity_df):
    if (35<=activity_df["activity_rate"] < 45):
        risk = 1
    elif (45<=activity_df["activity_rate"] < 55):
        risk = 2
    elif (55<=activity_df["activity_rate"] <= 65):
        risk = 3
    return int(risk)

def obesity_classification(activity_df):
    if (7.5 <= activity_df["percentage_obese"] < 12.5):
        risk = 1 
    elif (12.5<=activity_df["percentage_obese"] < 17.5):
        risk = 2 
    elif (17.5<=activity_df["percentage_obese"] < 22.5):
        risk = 3
    elif (22.5<=activity_df["percentage_obese"] <= 27.5):
        risk = 4 
    return int(risk)

def classify_distance(activity_df):
    if (0<=activity_df["distance_to_melbourne_km"] < 10):
        risk = 1
    elif (10<=activity_df["distance_to_melbourne_km"] < 20):
        risk = 2
    elif (20<=activity_df["distance_to_melbourne_km"] < 30):
        risk = 3
    elif (30<=activity_df["distance_to_melbourne_km"] <40):
        risk = 4
    elif (40<=activity_df["distance_to_melbourne_km"] < 50):
        risk = 5
    elif (50<=activity_df["distance_to_melbourne_km"] <=60):
        risk = 6
    return int(risk)

universal_df = pd.read_csv("obesity_merged_stats.csv", encoding = "ISO-8859-1")
health_data = pd.read_csv("health_loc.csv", encoding = 'ISO-8859-1')
activity_rate = health_data[[" lga_name", " ppl_not_meet_phys_activity_glines_perc", " remoteness_area"]]
general_df = pd.merge(left=universal_df, right=activity_rate, left_on="lga_name", right_on=" lga_name").drop(columns = " lga_name").rename(columns = {" lga_area_km2": "lga_area_km2"," ppl_not_meet_phys_activity_glines_perc": "low_activity_rate"})
lga_names = general_df['lga_name'].apply(remove_brackets)
general_df['lga_name'] = lga_names.astype(str)
general_df = general_df.rename(columns = {"lga_name":"LGA"})
# general_df_open
general_df['p_density'] = general_df["population_2015"]/general_df["lga_area_km2"]
general_df
# discretise the distance to melbourne into 3 bins (for colour coding and mutual information analysis)
#Create a list containing labels for distance range to Melbourne
distance = ['Within 10km to CBD', 'Within 20km to CBD', 'Within 30km to CBD', 'Within 40km to CBD', 'Within 50km to CBD', 'Within 60km to CBD']
distance_index = 0
general_df["activity_rate"] = 100 - general_df['low_activity_rate']
columns = [' remoteness_area', 'obesity rate']

# Scatter plot for LGA, activity_rate and obesity rate
# clustering
# regression
# and distance to melbourne

#LGA_activity_obesity = general_df[['LGA', "percentage_obese", 'low_activity_rate']]
LGA_activity_obesity = general_df[['LGA', "percentage_obese", 'activity_rate','distance_to_melbourne_km']]
#scatter_plot = plt.scatter(LGA_activity_obesity.iloc[:,1],LGA_activity_obesity.iloc[:,2],s=30)
#plt.xticks(rotation = 45)
#clustering(LGA_activity_obesity[["low_activity_rate","percentage_obese"]],3)
clustering(LGA_activity_obesity[["activity_rate","percentage_obese"]],3)
for i, txt in enumerate(LGA_activity_obesity['LGA']):
    plt.annotate(txt, (LGA_activity_obesity['activity_rate'][i], LGA_activity_obesity['percentage_obese'][i]))
#Resize scatter plot
plt.gcf().set_size_inches((17, 17)) 
plt.xlabel("Population Active (%)")
plt.ylabel('Population Obese (%)')
plt.title("Activity Rates and Obesity")
plt.grid(True)

plt.savefig("clustering for obesity rate and activity rate.png")
plt.show()

# Running linear regression for low activity rates and obesity rates

LGA_activity_obesity.corr()
low_activity_test = ([35,30,25])
low_activity = pd.DataFrame(LGA_activity_obesity["activity_rate"]) 
obesity_rates = pd.DataFrame(LGA_activity_obesity["percentage_obese"])
dependent = obesity_rates
independent = low_activity

#Train test split

X_train,X_test, y_train, y_test= train_test_split(independent, dependent, train_size = 0.80, test_size=0.20, random_state=42)

# build linear regression model

lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
m = round(float(model.coef_),2)
c = round(float(model.intercept_),2)
print(f"Equation of the regression line: y = {m}x+{c}")
model.score(X_train,y_train) # R^2 value score
LGA_activity_obesity.plot(kind='scatter', x='activity_rate', y='percentage_obese')
plt.plot(independent,model.predict(independent), color = 'red', linewidth = 2,label='y={:.2f}x+{:.2f}'.format(m,c))
plt.legend(fontsize=9)
plt.xlabel("Population Active (%)")
plt.ylabel('Population Obese (%)')
plt.title("Activity Rates and Obesity")
plt.grid(True)
plt.savefig("Linear Regression for obesity rate and activity rate.png")
plt.show()



# distance to melbourne plot 
LGA_activity_obesity['colour'] = LGA_activity_obesity.apply(colour_code, axis=1)
LGA_activity_obesity  = LGA_activity_obesity [["LGA","percentage_obese", 'activity_rate', "colour","distance_to_melbourne_km"]].sort_values("distance_to_melbourne_km")
distance = ['Within 10km to CBD', 'Within 20km to CBD', 'Within 30km to CBD', 'Within 40km to CBD', 'Within 50km to CBD', 'Within 60km to CBD']
distance_index = 0
LGA_activity_obesity.reset_index(drop=True, inplace=True)

for index, row in LGA_activity_obesity.iterrows():
    if index == 0:
        plt.scatter(LGA_activity_obesity.iloc[index, 2], LGA_activity_obesity.iloc[index, 1], color=LGA_activity_obesity.iloc[index, 3], label=distance[distance_index])
    elif (LGA_activity_obesity.iloc[index, 3] == LGA_activity_obesity.iloc[index-1, 3]):
        plt.scatter(LGA_activity_obesity.iloc[index, 2], LGA_activity_obesity.iloc[index, 1], color=LGA_activity_obesity.iloc[index, 3])
    else:
        distance_index += 1
        plt.scatter(LGA_activity_obesity.iloc[index, 2], LGA_activity_obesity.iloc[index, 1], color=LGA_activity_obesity.iloc[index, 3], label=distance[distance_index])

for i, txt in enumerate(LGA_activity_obesity['LGA']):
    plt.annotate(txt, (LGA_activity_obesity['activity_rate'][i], LGA_activity_obesity['percentage_obese'][i]))
#Resize scatter plot
plt.gcf().set_size_inches((17, 17)) 
plt.legend()
plt.xlabel("Population Active (%)")
plt.ylabel('Population Obese (%)')
plt.title("Activity Rates and Obesity")
plt.grid(True)
plt.savefig("obesity rate and activity rate.png",bbox_inches = 'tight')
plt.show()
print(f"The linear regression Score(R^2): {round(model.score(low_activity, obesity_rates),2)}")


# Agglomerative Hierarchical Cluster map  for percentage obese and activity rate

points = general_df[["percentage_obese", "activity_rate","LGA"]]
points = points.rename(columns = {"percentage_obese": "Population Obese (%)", "activity_rate": "Population Active (%)"})
points = points.set_index("LGA")

# performing agglomerative clustering

sns.clustermap(points, cmap = 'coolwarm')
plt.title("Heatmap for Open Spaces, Activity Rate and Obesity", size = 15, loc = 'center')
plt.savefig("clustermap for open spaces.png", bbox_inches ='tight')
plt.show()

general_df["activity class"] = general_df.apply(classify_activity, axis = 1)
general_df["obesity class"] = general_df.apply(obesity_classification, axis = 1)

# filtering variables from the general dataframe and renaming them
df = general_df[["obesity class","activity class"]]
dependent = df["obesity class"]
independent = df["activity class"]

# Train test split for Mutual Information

X_train,X_test, y_train, y_test= train_test_split(dependent, independent, test_size=0.20, random_state=42)
# mutual information values
mutual_info = mutual_info_score(X_train, y_train)
print(f"Mutual Information Score for Obesity and Activity Rate: {round(mutual_info,2)}")