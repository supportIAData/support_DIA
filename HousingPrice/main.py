import pandas as pd
import numpy as np
import warnings
import datetime 

warnings.filterwarnings("ignore")
### Data description
### Data Cleaning


class GetDataFrame:
    def __init__(self, dfname="train"):
        if dfname == "train":
            self.df = pd.read_csv("dataset/train.csv")
        else:
            self.df = pd.read_csv("dataset/test.csv")

        self.numerical_features = list()
        self.categorical_features = list()
        self.timeseries_features = list()
        self.feature_outlier_count = {}
        self.house_price = {}

        # self.shape = self.df.shape
        # self.info = self.df.info()

    def get_cleaned_df(self):
        self.df.drop("Id", axis=1, inplace=True)
        NA_columns = [
            "Alley",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "PoolQC",
            "Fence",
            "MiscFeature",
        ]
        self.df[NA_columns] = self.df[NA_columns].fillna("NA")
        self.df["LotFrontage"] = self.df.groupby(["Neighborhood", "LotConfig"])[
            "LotFrontage"
        ].apply(lambda x: np.Nan if x.median() == np.NaN else x.fillna(x.median()))
        self.df["LotFrontage"] = self.df.groupby(["LotConfig"])["LotFrontage"].apply(
            lambda x: x.fillna(x.median())
        )
        self.df.loc[self.df.GarageYrBlt.isnull(), "GarageYrBlt"] = self.df.loc[
            self.df.GarageYrBlt.isnull(), "YearBuilt"
        ]
        # fill 0 and Not Present for numerical and categorical feature's null values
        self.df.MasVnrArea.fillna(0, inplace=True)
        self.df.MasVnrType.fillna("Not present", inplace=True)
        # Adding square feet of first floor and second floor
        self.df["TotalFlrSFAbvGrd"] = self.df[["1stFlrSF", "2ndFlrSF"]].sum(axis=1)
        # Adding all the bathrooms
        self.df["TotalBath"] = self.df[
            ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]
        ].sum(axis=1)
        # Adding square feet of all Porcch
        self.df["TotalPorchSF"] = self.df[
            ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        ].sum(axis=1)
        self.df.drop(columns=self.get_columns_to_drop(), inplace=True)
        # create a list of numerical features
        self.numerical_features = list(
            self.df.select_dtypes(include=[np.number]).columns.values
        )
        # create a list of features that are categorical
        self.categorical_features = list(
            self.df.select_dtypes(include=[np.object]).columns.values
        )
        # Create feature list for time series data
        self.timeseries_features = [
            "YearBuilt",
            "YearRemodAdd",
            "YrSold",
            "MoSold",
            "GarageYrBlt",
        ]
        # removing times series features from numeric to avoid repetition
        for col in self.timeseries_features:
            self.numerical_features.remove(col)
        ###### Numerical #######
        # adding numerical features to categorical if the unique value count in a feature is less tha are equal to 10
        self.cat_feature = (
            pd.Series(
                self.df[self.numerical_features].nunique().sort_values(), name="Count"
            )
            .to_frame()
            .query("Count <= 10")
            .index.values
        )
        self.categorical_features.extend(self.cat_feature)
        # removing the numerical features that belong to time series
        for col in self.cat_feature:
            self.numerical_features.remove(col)

        # Handle outliers
        self.fix_outliers()

        # drop the variables that are not in correlation with sale price
        self.df.drop(["BsmtFinSF2", "BsmtUnfSF", "EnclosedPorch"], axis=1, inplace=True)
        for col in ["BsmtFinSF2", "BsmtUnfSF", "EnclosedPorch"]:
            self.numerical_features.remove(col)

        ######### Categorical ##########

        # change the type to string
        self.df.MSSubClass = self.df.MSSubClass.astype(str)

        # reducing the number of categories
        self.df.MSSubClass.replace(
            {
                "20": "1story",
                "30": "1story",
                "40": "1story",
                "45": "1story",
                "50": "1story",
                "60": "2story",
                "70": "2story",
                "75": "2story",
                "80": "nstory",
                "85": "nstory",
                "90": "nstory",
                "120": "1story",
                "150": "1story",
                "160": "2story",
                "180": "nstory",
                "190": "nstory",
            },
            inplace=True,
        )
        # adding MSSubClass to categorical Feature list
        self.categorical_features.append("MSSubClass")
        # removing it from numerical feature list
        self.numerical_features.remove("MSSubClass")
        # Combine Categories that are not ordinal as ordinal catgeories (need refactorization)
        self.df.BldgType.replace({"2fmCon": "Twnhs", "Duplex": "Twnhs"}, inplace=True)
        self.df.BsmtExposure.replace({"Mn": "Av"}, inplace=True)
        self.df.Condition1.replace(
            {"RRNn": "RRAn", "PosN": "PosA", "RRNe": "RRAe", "Feedr": "Artery"},
            inplace=True,
        )
        self.df.Exterior2nd.replace(
            {
                "MetalSd": "Wd Sdng",
                "Wd Shng": "Wd Sdng",
                "HbBoard": "Wd Sdng",
                "Plywood": "Wd Sdng",
                "Stucco": "Wd Sdng",
                "CBlock": "BrkFace",
                "Other": "BrkFace",
                "Stone": "BrkFace",
                "AsphShn": "BrkFace",
                "ImStucc": "BrkFace",
                "Brk Cmn": "BrkFace",
            },
            inplace=True,
        )
        self.df.Foundation.replace({"Wood": "Stone", "Slab": "Stone"}, inplace=True)
        self.df.GarageType.replace(
            {
                "CarPort": "Detchd",
                "No Garage": "Detchd",
                "Basment": "Detchd",
                "2Types": "Detchd",
            },
            inplace=True,
        )
        self.df.LotShape.replace({"IR3": "IR2"}, inplace=True)
        self.df.MSZoning.replace({"RH": "RM"}, inplace=True)
        self.df.MasVnrType.replace(
            {"None": "BrkCmn", "Not present": "BrkCmn"}, inplace=True
        )
        self.df.Neighborhood.replace(
            {
                "BrDale": "MeadowV",
                "IDOTRR": "MeadowV",
                "NAmes": "Sawyer",
                "NPkVill": "Sawyer",
                "Mitchel": "Sawyer",
                "SWISU": "Sawyer",
                "Blueste": "Sawyer",
                "Blmngtn": "Gilbert",
                "SawyerW": "Gilbert",
                "NWAmes": "Gilbert",
                "ClearCr": "Crawfor",
                "CollgCr": "Crawfor",
                "Timber": "Veenker",
                "Somerst": "Veenker",
                "Edwards": "OldTown",
                "BrkSide": "OldTown",
                "StoneBr": "NridgHt",
                "NoRidge": "NridgHt",
            },
            inplace=True,
        )

        self.df.SaleCondition.replace(
            {"AdjLand": "Abnorml", "Alloca": "Abnorml", "Family": "Abnorml"},
            inplace=True,
        )
        self.df.SaleType.replace(
            {
                "ConLD": "COD",
                "ConLI": "COD",
                "CwD": "COD",
                "ConLw": "COD",
                "Con": "COD",
                "Oth": "COD",
            },
            inplace=True,
        )
        categorical_to_drop = [
            "ExterCond",
            "Fence",
            "LotConfig",
            "RoofStyle",
            "Exterior1st",
        ]
        # add columns to drop
        drop_columns = ["ExterCond", "Fence", "LotConfig", "RoofStyle", "Exterior1st"]

        # drop the selected features
        self.df.drop(columns=drop_columns, inplace=True)

        # remove the dropped columns from categorical feature list
        for cat in drop_columns[:]:
            self.categorical_features.remove(cat)

        ######### Timeseries ########
        # change the types to integer
        self.df.YrSold = self.df.YrSold.astype(int)
        self.df.GarageYrBlt = self.df.GarageYrBlt.astype(int)

        # create a derieved column date sold by combining month sold and year sold
        self.df["dateSold"] = (
            self.df["MoSold"].astype(str) + "-1-" + self.df["YrSold"].astype(str)
        )
        self.df["dateSold"] = pd.to_datetime(self.df["dateSold"])
        # add the new column to the timeseries list
        self.timeseries_features.append("dateSold")

        # drop des colonnes Mosold / Yrsold
        self.df.drop(["MoSold", "YrSold"], axis=1, inplace=True)
        for col in ["MoSold", "YrSold"]:
            self.timeseries_features.remove(col)

        # reset these categorical numerical variables to integer
        self.df[
            [
                "HalfBath",
                "Fireplaces",
                "FullBath",
                "BsmtFullBath",
                "GarageCars",
                "BedroomAbvGr",
                "OverallCond",
                "OverallQual",
            ]
        ] = self.df[
            [
                "HalfBath",
                "Fireplaces",
                "FullBath",
                "BsmtFullBath",
                "GarageCars",
                "BedroomAbvGr",
                "OverallCond",
                "OverallQual",
            ]
        ].astype(
            int
        )

        # assign the categorical columns that are non integer to categorical_columns as a list
        self.categorical_columns = [
            "ExterQual",
            "BsmtQual",
            "BsmtCond",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "HouseStyle",
            "BsmtFinType2",
            "BsmtFinType1",
            "GarageFinish",
        ]

        # assign the labels in the order of decreasing to increasing as when creating a categorical feature
        # Converting normal Object features to Categorical Data Type features
        self.df["ExterQual"] = pd.Categorical(
            self.df["ExterQual"], ordered=True, categories=["Fa", "TA", "Gd", "Ex"]
        )
        self.df["BsmtQual"] = pd.Categorical(
            self.df["BsmtQual"], ordered=True, categories=["NA", "Fa", "TA", "Gd", "Ex"]
        )
        self.df["BsmtCond"] = pd.Categorical(
            self.df["BsmtCond"], ordered=True, categories=["NA", "Po", "Fa", "TA", "Gd"]
        )
        self.df["HeatingQC"] = pd.Categorical(
            self.df["HeatingQC"], ordered=True, categories=["Po", "Fa", "TA", "Gd", "Ex"]
        )
        self.df["KitchenQual"] = pd.Categorical(
            self.df["KitchenQual"], ordered=True, categories=["Fa", "TA", "Gd", "Ex"]
        )
        self.df["FireplaceQu"] = pd.Categorical(
            self.df["FireplaceQu"],
            ordered=True,
            categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        )
        self.df["GarageQual"] = pd.Categorical(
            self.df["GarageQual"],
            ordered=True,
            categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        )
        self.df['GarageFinish'] = pd.Categorical(self.df['GarageFinish'],ordered=True,categories=['NA','Unf','RFn','Fin'])
        self.df['BsmtFinType1']=pd.Categorical(self.df['BsmtFinType1'],ordered=True,categories=['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'])
        self.df['BsmtFinType2']=pd.Categorical(self.df['BsmtFinType2'],ordered=True,categories=['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'])
        self.df['HouseStyle']=pd.Categorical(self.df['HouseStyle'],ordered=True,categories=[ 'SFoyer','1.5Unf','1Story','1.5Fin','SLvl','2.5Unf','2Story','2.5Fin'])

        # factorize the categories to Integer representation
        for col in self.categorical_columns:
            code, _ = pd.factorize(self.df[col],sort=True)
            self.df[col] = pd.Series(code)

        # reassign the categorical features 
        self.categorical_features = list(self.df.select_dtypes(include=[np.object]).columns.values)

        # created dummy variables for categorical features
        self.house_price = pd.concat([self.df,pd.get_dummies(self.df[self.categorical_features],drop_first=True)],axis=1)

        # drop the actual categorical feature from list
        self.house_price.drop(columns=self.categorical_features,inplace=True)
        # reset index for the new dataframe
        self.house_price.reset_index(drop=True,inplace=True)

        ######## TIME ########
        # lets create a constant time
        self.tm = datetime.time(10,10)

        # convert the dateSold to unixstimestamp
        self.house_price.dateSold = self.house_price.dateSold.apply(lambda x: datetime.datetime.combine(x, self.tm).timestamp())

        # reassigning all the numerical features to the numerical_features variable as a list
        self.numerical_features = list(self.df.select_dtypes(include=[np.number]).columns.values)
        print(self.house_price.shape)
        return self.df

    def get_raw_df(self):
        return self.df

    def check_null_percentage(self):
        missing_info = (
            pd.DataFrame(
                np.array(
                    self.df.isnull().sum().sort_values(ascending=False).reset_index()
                ),
                columns=["Columns", "Missing_Percentage"],
            )
            .query("Missing_Percentage > 0")
            .set_index("Columns")
        )
        return 100 * missing_info / self.df.shape[0]

    def get_columns_to_drop(self):
        # get the features that holds more than 90% of its data with a same value
        unique_df = (
            self.df.apply(lambda x: self.top_unique_count(x))
            .rename(index={0: "Value", 1: "Percentage", 2: "Count"})
            .T.sort_values(by="Count", ascending=False)
        )
        drop_columns = unique_df.query("Percentage > 90.0").index.values
        return drop_columns

    def top_unique_count(self, x):
        unq_cnt = (
            x.value_counts(ascending=False, dropna=False).head(1).index.values[0],
            100
            * x.value_counts(ascending=False, dropna=False).head(1).values[0]
            / self.df.shape[0],
            x.value_counts(ascending=False, dropna=False).head(1).values[0],
        )
        return unq_cnt

    def feature_outlier_make_count(self, to_print=False):
        self.feature_outlier_count = {
            "1stFlrSF": 1,
            "BsmtFinSF1": 2,
            "BsmtFinSF2": 1,
            "EnclosedPorch": 2,
            "GarageArea": 4,
            "GrLivArea": 4,
            "LotArea": 7,
            "LotFrontage": 2,
            "MasVnrArea": 1,
            "OpenPorchSF": 3,
            "TotalBsmtSF": 4,
            "TotRmsAbvGrd": 1,
            "TotalFlrSFAbvGrd": 2,
            "TotalPorchSF": 1,
            "WoodDeckSF": 3,
        }

        if to_print == True:
            for k, v in self.feature_outlier_count.items():
                if v:
                    print(
                        self.df.loc[
                            self.df[k].isin(sorted(self.df[k])[-v:]), [k, "SalePrice"]
                        ]
                    )

    def get_outliers(self, feature, index=-1):
        outlier = self.df.loc[
            self.df[feature] == sorted(self.df[feature])[index], [feature, "SalePrice"]
        ].sort_values(by=feature, ascending=False)
        return outlier

    def remove_outlier_features_count_for_index(self, outlier_index):
        for col in self.feature_outlier_count.keys():
            if (self.feature_outlier_count[col] > 0) & (
                outlier_index in self.get_outliers(col).index.values
            ):
                self.feature_outlier_count[col] = self.feature_outlier_count[col] - 1
        self.df.drop(outlier_index, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return "ok"

    def fix_outliers(self):
        self.feature_outlier_make_count()
        for k, v in self.feature_outlier_count.items():
            while v > 0:
                # replacing the outliers by taking mean of four closest feature value of the outlier at the salePrice Range
                replace_with = self.df.loc[
                    (self.df["SalePrice"] - self.get_outliers(k)["SalePrice"].values[0])
                    .abs()
                    .argsort()[v : v + 4],
                    k,
                ].mean()
                if (self.df[k].dtypes == np.int64) | (self.df[k].dtypes == np.int32):
                    self.df.loc[
                        self.df.index[self.get_outliers(k).index.values[0]], k
                    ] = int(replace_with)
                else:
                    self.df.loc[
                        self.df.index[self.get_outliers(k).index.values[0]], k
                    ] = round(replace_with, 1)
                v = v - 1
                self.feature_outlier_count[k] = v


df = GetDataFrame("train")
toto = df.get_cleaned_df()
# print(toto.head)
# print(toto)
