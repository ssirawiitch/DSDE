#import your other libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, file_path): # DO NOT modify this line
        #Add other parameters if needed
        self.file_path = file_path 
        self.df = None #parameter for loading csv

    def Q1(self): # DO NOT modify this line
        """
        Step1-4
            1. Load the CSV file.
            2. Choose edible mushrooms only.
            3. Only the variables below have been selected to describe the distinctive
               characteristics of edible mushrooms:
               'cap-color-rate','stalk-color-above-ring-rate'
            4. Provide a proper data preprocessing as follows:
                - Fill missing with mean
                - Standardize variables with Standard Scaler
        """
        # remove pass and replace with you code
        df = pd.read_csv(self.file_path)
        edible_mushrooms = df[df['label'] == 'e']
        selected_features = edible_mushrooms[['cap-color-rate', 'stalk-color-above-ring-rate']]
        filled_data = selected_features.fillna(selected_features.mean())

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(filled_data)
        self.df = pd.DataFrame(standardized_data, columns=selected_features.columns)

        return self.df.shape

    def Q2(self): # DO NOT modify this line
        """
        Step5-6
            5. K-means clustering with 5 clusters (n_clusters=5, random_state=0, n_init='auto')
            6. Show the maximum centroid of 2 features ('cap-color-rate' and 'stalk-color-above-ring-rate') in 2 digits.
        """
        # remove pass and replace with you code
        df = pd.read_csv(self.file_path)
        edible_mushrooms = df[df['label'] == 'e']
        selected_features = edible_mushrooms[['cap-color-rate', 'stalk-color-above-ring-rate']]
        filled_data = selected_features.fillna(selected_features.mean())

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(filled_data)
        self.df = pd.DataFrame(standardized_data, columns=selected_features.columns)

        kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
        kmeans.fit(self.df)

        centroids = kmeans.cluster_centers_
        max_val = np.max(centroids, axis=0)

        return np.round(max_val,2)

    def Q3(self): # DO NOT modify this line
        """
        Step7
            7. Convert the centroid value to the original scale, and show the minimum centroid of 2 features in 2 digits.

        """
        # remove pass and replace with you code
        df = pd.read_csv(self.file_path)
        edible_mushrooms = df[df['label'] == 'e']
        selected_features = edible_mushrooms[['cap-color-rate', 'stalk-color-above-ring-rate']]
        filled_data = selected_features.fillna(selected_features.mean())

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(filled_data)
        self.df = pd.DataFrame(standardized_data, columns=selected_features.columns)

        kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
        kmeans.fit(self.df)

        centroids = kmeans.cluster_centers_
        original_centroid = scaler.inverse_transform(centroids)
        return np.round(np.min(original_centroid, axis=0), 2)