import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# Read files
obesity_df = pd.read_csv('new2_universal_df.csv', encoding='utf-8')

obesity_df = obesity_df.sort_values( by = 'CBD_distance_bin')

# Obtain sub-entries that contain relevant data for each continent
plot_array = []
distance_array = obesity_df.CBD_distance_bin.unique()
for i in range(0, len(distance_array)):
    plot_array.append(obesity_df.loc[obesity_df['CBD_distance_bin']==str(distance_array[i])])
    
colors = ['green', 'yellow', 'orange', 'red', 'purple', 'black']

plt.figure("legend_figure")


for j,c,l in zip(plot_array, colors, distance_array): 
    plt.scatter(j.iloc[:,6], j.iloc[:,1], s=40, color=c, label=l)
    
obesity_df = obesity_df.sort_values( by = 'lga_name')
    
#Annotate each scatter point with respective lga name
for i, txt in enumerate(obesity_df['lga_name']):
    text = plt.annotate(txt, (obesity_df['gyms_per_km2'][i], obesity_df['percentage_obese'][i]))
    text.set_fontsize(10)
    
#Resize scatter plot
plt.gcf().set_size_inches((17, 17)) 

plt.title('Proportion of Gyms to Obesity Rate', fontsize=20)
plt.xlim(-0.005, 0.42)
plt.ylim(7.5, 27.5)
plt.ylabel("Population Obese (%)", fontsize=15)
plt.xlabel("Proportion of Gyms (No. gyms / km2)", fontsize=15)
plt.legend(prop={'size': 15})

plt.savefig("proportion_gyms to obesity (lga map).png", bbox_inches='tight')




#Apply Pearson's correlation

new_obesity = obesity_df[['percentage_obese', 'gyms_per_km2']]
correlation_score = new_obesity.corr(method='pearson', min_periods=1)
print("Pearsons score")
print(correlation_score)
print("")



# Mutual information

def obesity_classification(obesity_df):
    if (7.5 <= obesity_df["percentage_obese"] < 12.5):
        risk = '1'
    elif (12.5<=obesity_df["percentage_obese"] < 17.5):
        risk = '2'
    elif (17.5<=obesity_df["percentage_obese"] < 22.5):
        risk = '3'
    elif (22.5<=obesity_df["percentage_obese"] <= 27.5):
        risk = '4'
    return risk

# Classifying the obesity rates low = (7.5-12.5) medium = (12.5 - 17.5), normal = (17.5 - 22.5), high = (22.5 - 27.5)
obesity_df["obesity class"] = obesity_df.apply(obesity_classification, axis = 1)
# filtering variables from the general dataframe and renaming them
df = obesity_df[["obesity class", "lga_area_km2", "population_2015", "distance_to_melbourne_km", "gyms_per_km2"]]
# Test train split to prevent overfitting
'''
X_train,X_test, y_train, y_test= train_test_split(df.drop(labels = ['obesity class'], axis = 1), df['obesity class'], train_size=0.80, test_size=0.20, random_state=42)
# information gain
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print("Mutual Information Scores")
print(mutual_info.sort_values(ascending = False))
print("")
'''


#Plot data points with linear regression line
obesity_df.corr()
independent_var = pd.DataFrame(obesity_df['gyms_per_km2'])
dependent_var = pd.DataFrame(obesity_df['percentage_obese'])
# Training split
X_train,X_test, y_train, y_test= train_test_split(independent_var, dependent_var, train_size=0.80, test_size=0.20, random_state=42)
#Build linear regression model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
model.coef_
model.intercept_
model.score(independent_var, dependent_var)
print(f'R^2 score: {round(model.score(independent_var, dependent_var),2)}') #R^2 value score
#Print the regression formula
m = round(float(model.coef_),2)
c = round(float(model.intercept_),2)
print(f"Equation of the regression line: y = {m}x+{c}")
obesity_df.plot(kind='scatter', x='gyms_per_km2', y='percentage_obese')
plt.plot(independent_var, model.predict(independent_var), color = 'red', linewidth = 1, label='y={:.2f}x+{:.2f}'.format(m,c))
plt.legend(fontsize=9)
#Limit x and y axis
#plt.xlim(3.5, 21.5)
#plt.ylim(6.5, 29)
#Label the x and y-axis for the scatter plot
plt.xlabel('Proportion of Gyms (No. gyms / km2)')
plt.ylabel('Population Obese (%)')
#Create a title
plt.title('Proportion of Gyms to Obesity Rate')
plt.savefig("proportion_gyms to obesity (regression).png", bbox_inches='tight')
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
   
# Cluster data points in scatter plot for percentage of population that consumes soft drink vs percentage obese
clustering(obesity_df[['gyms_per_km2', 'percentage_obese']], 4)
#Annotate each scatter point with respective lga name
for i, txt in enumerate(obesity_df['lga_name']):
    text = plt.annotate(txt, (obesity_df['gyms_per_km2'][i], obesity_df['percentage_obese'][i]))
    text.set_fontsize(10)
    
#Resize scatter plot
plt.gcf().set_size_inches((17, 17))

#Limit x and y axis
plt.xlim(-0.005, 0.42)
plt.ylim(7.5, 27.5)
#Label the x and y-axis for the scatter plot
plt.xlabel('Proportion of Gyms (No. gyms / km2)', fontsize=15)
plt.ylabel('Population Obese (%)', fontsize=15)
#Create a title for the scatter plot
plt.title('Proportion of Gyms to Obesity Rate', fontsize=20)
plt.savefig("proportion_gyms to obesity (clusters).png", bbox_inches='tight')
plt.show()