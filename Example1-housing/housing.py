import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        min_val = np.min(X[:, bedrooms_ix])
        max_val = np.max(X[:, bedrooms_ix])
        bedrooms_per_room = (X[:, bedrooms_ix] - min_val) / (max_val - min_val)

        if self.add_bedrooms_per_room:
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


#Function check whether the id is in the hash table before 
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

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

def load_data(file):
    df = pd.read_csv(file)
    return df


def main():
    dataframe = load_data("housing.csv")

    #A glance on the data set
    #print(dataframe["ocean_proximity"].value_counts())
    #print(dataframe.describe())
    #dataframe.hist(bins=50, figsize=(20,15))
    #housing["income_cat"].hist()

    ####################################################################################################

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

    ####################################################################################################

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
        #set_.drop("ocean_proximity", axis=1, inplace=True)
        
    ####################################################################################################
    
    #Visualize andd discover data to gain insights
    #Create a copy of the dataframe, analysis it without affect the original dataframe
    #housing = strat_train_set.copy()
    #housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    #housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
    
    #plt.legend()
    #plt.show()

    ####################################################################################################

    #Look for correlation
    #Check the linear relationship between variable (between median_house_value and oher 4 traits)
    #corr_matrix = housing.corr()
    #print(corr_matrix["median_house_value"].sort_values(ascending=False))

    #attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    #scatter_matrix(housing[attributes], figsize=(12,8))
    #plt.show()

    #housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
    #plt.show()

    ####################################################################################################

    #Experiment with Attribute combinations
    # Create more features which make more sense in terms of liner relationship
    #housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    #housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    #housing["population_per_household"]=housing["population"]/housing["households"]

    #corr_matrix = housing.corr()
    #print(corr_matrix["median_house_value"].sort_values(ascending=False))
    #Output of the bedrooms_per_room has the standard corelation coefiicient -0.25 = much more correlated with the median house value than the total number of rooms or bedrooms

    # Create a copy    
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    #print(housing)
    #print(housing_labels)

    ####################################################################################################

    #Prepare/Clean data
    # 3 options
    #housing.dropna(subset=["total_bedrooms"]) # option 1 Get rid of the corresponding districts/missing one.
    #housing.drop("total_bedrooms", axis=1) # option 2 Get rid of the whole attribute.
    #median = housing["total_bedrooms"].median() # option 3 Set the values to some value (zero, the mean, the median, etc.)
    #housing["total_bedrooms"].fillna(median, inplace=True)

    #Scikit-Learn provides a handy class to take care of missing values: SimpleImputer
    # imputer = SimpleImputer(strategy="median")
    # housing_num = housing.drop("ocean_proximity", axis=1) #median can only be computed on numerical attributes, you need to create a copy of the data without the text attribute ocean_proximity
    # imputer.fit(housing)
    # print(imputer.statistics_)
    # print(housing.median().values)
    # X = imputer.transform(housing) # replacing missing values with the learned medians, but in Numpy array
    # print(X)
    # housing_tr = pd.DataFrame(X, columns=housing.columns, index=housing.index) # Tranform form numpy array to dataframe pandas
    # print(housing_tr)

    ####################################################################################################

    #Handling text and categories attribute
    #Check the type of text
    #Convert the non-numerical value into numerical value
    # Issue: The transformation is a problem since the algorithm in ML will similiaring the near value 
    # This transformation is good for bad, moderate and good but in this case not good 
    housing_cat = housing[["ocean_proximity"]] # Transform into 2D array beacuse ordinal_encoder.fit_transform function accept the 2D array
    housing_num = housing.drop("ocean_proximity", axis=1)
    # print(housing_cat.head())
    # ordinal_encoder = OrdinalEncoder()
    # housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    # print(housing_cat_encoded[:10])

    # Solution for the problem above is one-hot encoding 
    # one attribute equal to 1 when the cate‐gory is “<1H OCEAN” (and 0 otherwise), 
    # another attribute equal to 1 when the cate‐gory is “INLAND” (and 0 otherwise), and so on.
    # one attribute will be equal to 1 (hot), while the others will be 0 (cold).
    # cat_encoder = OneHotEncoder()
    # housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # display in a SciPy sparse matrix
    # print(housing_cat_1hot.toarray())
    # print(cat_encoder.categories_)

    ####################################################################################################

    # Transformation Pipelines: Custom execute: the median value, feature scaling, add new attribute
    # This part handle the numerical attribute
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])
    # housing_num_tr = num_pipeline.fit_transform(housing_num)
    # print(housing_num_tr)

    ####################################################################################################

    #Scikit-Learn introduced the ColumnTransformer for handled the categorical columns and the numerical columns at the same time
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),])
    housing_prepared = full_pipeline.fit_transform(housing)
    #housing_tr = pd.DataFrame(housing_prepared, columns=housing.columns, index=housing.index)
    #print(housing_prepared[0])

    ####################################################################################################

    #Select a train model

    #Training and evaluating on the training set
    lin_reg_model = LinearRegression()
    # #housing_prepared typically refers to the prepared input features or independent variables for training the model. It should be a numerical dataset or a NumPy array 
    # #housing_labels refers to the corresponding target variable or dependent variable values for the training data. It should be a 1-dimensional array or a pandas Series 
    lin_reg_model.fit(housing_prepared, housing_labels)
    joblib.dump(lin_reg_model, "lin_reg_model.pkl") #save the model as "lin_reg_model"


    # #Check some data by the model
    # # some_data = housing.iloc[:5]
    # # some_labels = housing_labels.iloc[:5]
    # # some_data_prepared = full_pipeline.transform(some_data)
    # # print("Predictions:", lin_reg.predict(some_data_prepared))
    # # print("Labels:", list(some_labels))

    # #measure this regression model’s RMSE on the whole training set using Scikit-Learn’s mean_squared_error() function
    # housing_predictions = lin_reg.predict(housing_prepared)
    # #print(housing_predictions)
    # lin_mse = mean_squared_error(housing_labels, housing_predictions)
    # print(lin_mse)
    # lin_rmse = np.sqrt(lin_mse)
    # print(lin_rmse)
    #Error is too high, means underfitting, we need to replace by other model

    ####################################################################################################

    #Decision  Tree Regressor model 
    tree_reg_model = DecisionTreeRegressor()
    tree_reg_model.fit(housing_prepared, housing_labels)
    joblib.dump(tree_reg_model, "tree_reg_model.pkl") #save the model as "tree_reg_model"
    # housing_predictions = tree_reg.predict(housing_prepared)
    # tree_mse = mean_squared_error(housing_labels, housing_predictions)
    # tree_rmse = np.sqrt(tree_mse)
    # print(tree_rmse)
    #Error = 0, means overfitting, we need to evalution using cross validation 
    #Note that dont touch the test dataset until you are confident about traindata

    ####################################################################################################

    # Better Evaluation Using Cross-Validation
    # train_test_split() function to split the training set into a smaller training set and a validation set, 
    # then train your models against the smaller training set and evaluate them against the validation set
    # use Scikit-Learn’s K-fold cross-validation feature. The follow‐ing code randomly splits the training set into 10 distinct subsets called folds
    # then it trains and evaluates the Decision Tree model 10 times, picking a different fold for
    # evaluation every time and training on the other 9 folds. The result is an array con‐taining the 10 evaluation scores:
    # scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    # tree_rmse_scores = np.sqrt(-scores)
    #print(tree_rmse_scores)
    # Scikit-Learn’s cross-validation features expect a utility function
    # (greater is better) rather than a cost function (lower is better), so
    # the scoring function is actually the opposite of the MSE (i.e., a negative value), which is why the preceding code computes -scores before calculating the square root.
    #display_scores(tree_rmse_scores)
    #Output: Mean: 71407.68766037929, Standard deviation: 2439.4345041191004

    # lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
    # lin_rmse_scores = np.sqrt(-lin_scores) 
    #display_scores(lin_rmse_scores)
    # Mean: 69156.06808447083, Standard deviation: 2462.909908120931
    # the Decision Tree model is overfitting so badly that it performs worse than the Linear Regression model.

    ####################################################################################################

    # Random Forest Regressor 
    # forest_reg_model = RandomForestRegressor()
    # forest_reg_model.fit(housing_prepared, housing_labels)
    # joblib.dump(forest_reg_model, "forest_reg_model") #save the model as "forest_reg_model"
    # forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
    # forest_rmse_scores = np.sqrt(-forest_scores)
    # display_scores(forest_rmse_scores)

    ####################################################################################################

    # save every model you experiment with so that you can come back easily to any model you want
    # Make sure you save both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well
    # easily compare scores across model types, and compare the types of errors they make

    #Retrieve the models saved to compared cross-valuation 
    lin_reg_model_load = joblib.load("lin_reg_model.pkl")
    lin_scores = cross_val_score(lin_reg_model_load, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    tree_reg_model_load = joblib.load("tree_reg_model.pkl")
    tree_scores = cross_val_score(tree_reg_model_load, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    display_scores(tree_rmse_scores)

if __name__ == "__main__":
    main()