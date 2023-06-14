import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

#Function check whether the id is in the hash table before 
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    # result is a boolean Series where each element represents whether the corresponding identifier belongs to the test set (True) or not (False).
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def split_data_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def main():
    dataframe = load_data("housing.csv")

    #A glance on the data set
    #print(dataframe["ocean_proximity"].value_counts())
    #print(dataframe.describe())
    #dataframe.hist(bins=50, figsize=(20,15))
    #housing["income_cat"].hist()

    #Split the data set into train and test data set
    #train_data_set, test_data_set = split_data_test(dataframe, 0.2)
    #print(len(train_data_set))
    #print(len(test_data_set))

    #Split the data set into train and test data setwith category trait
    #housing_with_id = dataframe.reset_index() # adds an `index` column
    #train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    #print(len(train_set))
    #print(len(test_set))

    #train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)
    #print(len(train_set))
    #print(len(test_set))

    #Use Sckit leanr to divide the data set into groups with the same proportional to the orginal dataset
    dataframe["income_cat"] = pd.cut(dataframe["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    #dataframe["income_cat"].hist()
    #plt.show()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(dataframe, dataframe["income_cat"]):
        strat_train_set = dataframe.loc[train_index]
        strat_test_set = dataframe.loc[test_index]
    
    #print(strat_train_set)
    #print(strat_test_set)
    
    #print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    #print(dataframe["income_cat"].value_counts() / len(dataframe))

    #Remove the column added to clasify the groups trait and also the non-numerical data   
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
        set_.drop("ocean_proximity", axis=1, inplace=True)
        
    #Visualize andd discover data to gain insights
    #Create a copy of the dataframe, analysis it without affect the original dataframe
    housing = strat_train_set.copy()
    #housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    #housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
    
    #plt.legend()
    #plt.show()

    #Look for correlation
    #Check the linear relationship between variable (between median_house_value and oher 4 traits)
    #corr_matrix = housing.corr()
    #print(corr_matrix["median_house_value"].sort_values(ascending=False))

    #attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    #scatter_matrix(housing[attributes], figsize=(12,8))
    #plt.show()

    #housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
    #plt.show()

    #Experiment with Attribute combinations
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]
    


def load_data(file):
    df = pd.read_csv(file)
    return df


if __name__ == "__main__":
    main()