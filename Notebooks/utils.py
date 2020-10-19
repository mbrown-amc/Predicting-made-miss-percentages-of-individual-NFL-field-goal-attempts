import os
import pandas as pd

def get_data(data):
    """
    Loads the data for the project.
    :param data: String. The data to be loaded. "pbp" for play by play, "weather" for weather.
    """
    
    import os
    import pandas as pd
    pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    datadir = pardir + "\\Data"
    if data == "pbp":
        load = pd.read_csv((datadir + "\\NFL Play by Play 2009-2018 (v5).csv"))
        return load
    elif data == "weather":
        load = pd.read_csv((datadir + "\\spreadspoke_scores.csv"))
        return load
    else:
        print("Sorry, no such file to load")
        
def clean_data(type, data):
    """
    Prepares the data for the project.
    :param type: String. The data source being to be cleaned. "pbp" for play by play, "weather" for weather.
    :param data: DataFrame. The data being prepared. Should match the type.
    """
    import pandas as pd
    if type == "pbp":
        data = data[["posteam", "game_date", "half_seconds_remaining", "game_half", "field_goal_result", "kick_distance", "score_differential", "kicker_player_name", "kicker_player_id", "home_team"]].loc[data.play_type == "field_goal"][data.game_date > "2014-08-01"]
        data["game_date"] = pd.to_datetime(data["game_date"])
        data.loc[(data.posteam == "JAC"), "posteam"] = 'JAX'
        data.loc[(data.home_team == "JAC"), "home_team"] = 'JAX'
        data.loc[(data.field_goal_result == "blocked"), "field_goal_result"] = "missed"
    elif type == "weather":
        data["schedule_date"] = pd.to_datetime(data["schedule_date"])
        data = data[["schedule_date", "team_home", "stadium", "weather_temperature", "weather_wind_mph", "weather_detail"]].loc[data.schedule_date > "2014-08-01"]
        data.loc[(data.team_home == "Arizona Cardinals"), "team_home"] = 'ARI'
        data.loc[(data.team_home == "Atlanta Falcons"), "team_home"] = 'ATL'
        data.loc[(data.team_home == "Baltimore Ravens"), "team_home"] = 'BAL'
        data.loc[(data.team_home == "Buffalo Bills"), "team_home"] = 'BUF'
        data.loc[(data.team_home == "Carolina Panthers"), "team_home"] = 'CAR'
        data.loc[(data.team_home == "Chicago Bears"), "team_home"] = 'CHI'
        data.loc[(data.team_home == "Cincinnati Bengals"), "team_home"] = 'CIN'
        data.loc[(data.team_home == "Cleveland Browns"), "team_home"] = 'CLE'
        data.loc[(data.team_home == "Dallas Cowboys"), "team_home"] = 'DAL'
        data.loc[(data.team_home == "Denver Broncos"), "team_home"] = 'DEN'
        data.loc[(data.team_home == "Detroit Lions"), "team_home"] = 'DET'
        data.loc[(data.team_home == "Green Bay Packers"), "team_home"] = 'GB'
        data.loc[(data.team_home == "Houston Texans"), "team_home"] = 'HOU'
        data.loc[(data.team_home == "Indianapolis Colts"), "team_home"] = 'IND'
        data.loc[(data.team_home == "Jacksonville Jaguars"), "team_home"] = 'JAX'
        data.loc[(data.team_home == "Kansas City Chiefs"), "team_home"] = 'KC'
        data.loc[(data.team_home == "Los Angeles Chargers"), "team_home"] = 'LAC'
        data.loc[(data.team_home == "Los Angeles Rams"), "team_home"] = 'LA'
        data.loc[(data.team_home == "Miami Dolphins"), "team_home"] = 'MIA'
        data.loc[(data.team_home == "Minnesota Vikings"), "team_home"] = 'MIN'
        data.loc[(data.team_home == "New England Patriots"), "team_home"] = 'NE'
        data.loc[(data.team_home == "New Orleans Saints"), "team_home"] = 'NO'
        data.loc[(data.team_home == "New York Giants"), "team_home"] = 'NYG'
        data.loc[(data.team_home == "New York Jets"), "team_home"] = 'NYJ'
        data.loc[(data.team_home == "Oakland Raiders"), "team_home"] = 'OAK'
        data.loc[(data.team_home == "Philadelphia Eagles"), "team_home"] = 'PHI'
        data.loc[(data.team_home == "Pittsburgh Steelers"), "team_home"] = 'PIT'
        data.loc[(data.team_home == "San Diego Chargers"), "team_home"] = 'SD'
        data.loc[(data.team_home == "San Francisco 49ers"), "team_home"] = 'SF'
        data.loc[(data.team_home == "St. Louis Rams"), "team_home"] = 'STL'
        data.loc[(data.team_home == "Seattle Seahawks"), "team_home"] = 'SEA'
        data.loc[(data.team_home == "Tampa Bay Buccaneers"), "team_home"] = 'TB'
        data.loc[(data.team_home == "Tennessee Titans"), "team_home"] = 'TEN'
        data.loc[(data.team_home == "Washington Redskins"), "team_home"] = 'WAS'
        data = data.rename(columns = {"schedule_date":"game_date", "team_home":"home_team"})
        data["weather_detail"] = data.weather_detail.fillna("Normal")
    return data

from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

    
def ohe(CatVar, OtherVar, data, cat):
    """
    Performs One Hot Encoding.
    :param CatVar: List. The list of categorical variables in the data being passed.
    :param OtherVar: List. The non-categorical variables in the data being passed.
    :param data: DataFrame like object. The data to be encoded with OHE.
    :param cat: List. The list of possible values for each category, obtained with the get_cat function. Needed to make sure different encoding calls do not produce different numbers of columns.
    """
    
    encoder = OneHotEncoder(categories=cat)
    ColTrans = ColumnTransformer(transformers=[('ohe', encoder, CatVar)],remainder='passthrough', sparse_threshold = 0)
    trans = ColTrans.fit_transform(data)
    enc = ColTrans.named_transformers_['ohe']
    fnames = enc.get_feature_names(input_features = CatVar)
    cnames = list(fnames) + OtherVar
    return pd.DataFrame(trans, columns=cnames)

def get_cat(CatVar, data):
    """
    Gets the possible values for all categorical predictors, to be used with ohe. Necessary to avoid different numbers of columns when encoding.
    :param CatVar: List. The categorical variables in the data that will be passed.
    :param data: DataFrame like object. The full data for the project. Needed to ensure the models will never run into a value it does not know how to handle.
    """
    cat = []
    for var in CatVar:
        cat.append(data[var].unique())
    return cat
    
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
  
def cross_validate(X,y,model, kfold = "skfold", numsplits = 10, penalty = "l2", solver = "lbfgs", maxnodes = 100, minsamples = 200, estimators = 100, oob = True, maxfeatures = 'auto', maxdepth = 20, oversample = "None", ss = 1, score = "acc"):
    """
    Performs cross validation on passed data. Can handle multiple model types and parameters. Can also handle oversampling through RandomOverSampler.
    :param X: DataFrame like object. The X data being used for the cross validation.
    :param y: DataFrame like object. The y data being used for the cross validation.
    :param model: String. The type of model to be used. "LogReg" for logistic regression, "Tree" for decision tree, "Bag" for bagging, and "RF" for random forest.
    :param kfold: String. The type of kfold to use when cross validating. "skfold" for stratified, "kfold" for regular kfold.
    :param numsplits: Int. The number of folds to use for cross validation.
    :param penalty: String. The penalty to be used when cross validating. "l1" for l1, "l2" for l2.
    :param solver: String. The solver to use with cross validation.
    :param maxnodes: Int. The max number of nodes to be used when using a decision tree model.
    :param minsamples: Int. The minimum number of samples in each lead node when using a decision tree model.
    :param estimators: Int. The number of estimators to use for bagging and random forest models.
    :param oob: Boolean. Whether or not oob scores are used or not.
    :param maxfeatures: Variable. The max features that can be used in random forest models.
    :param maxdepth: Int. The max depth that can be used in random forest models.
    :param oversample: String. Whether oversampling will be performed during the cross validation. "None" for no oversampling, "ros" for RandomOverSampler.
    :param ss: Float. The sampling strategy (ratio of minority class to majority class) to use when oversampling.
    :param score: String. Which scoring method to use when cross validating. "acc" for classification accuracy, "f1" for f1.
    """
    tscores = []
    if kfold == "skfold":
        Kfold = StratifiedKFold(n_splits = numsplits)
    elif kfold == "kfold":
        Kfold = KFold(n_splits = numsplits)
    for train, test in Kfold.split(X,y):
        Xtrain = X.iloc[train]
        Xtest = X.iloc[test]
        ytrain = y[train]
        ytest = y[test]
        if oversample == "ros":
            print("ROS")
            ros = RandomOverSampler(random_state = 52594, sampling_strategy = ss)
            Xtrain, ytrain = ros.fit_resample(Xtrain, ytrain)
            print(Xtrain.shape)
        if model == "LogReg":
            Model = LogisticRegression(penalty = penalty, solver = solver, max_iter = 10000).fit(Xtrain, ytrain)
        elif model == "Tree":
            Model = tree.DecisionTreeClassifier(max_leaf_nodes=maxnodes, min_samples_leaf=minsamples)
            Model.fit(Xtrain, ytrain)
        elif model == "Bag":
            Model = BaggingClassifier(n_estimators = estimators, oob_score = oob).fit(Xtrain, ytrain)
        elif model == "RF":
            Model = RandomForestClassifier(n_estimators = estimators, oob_score = oob, max_features = maxfeatures, max_depth = maxdepth).fit(Xtrain, ytrain)
        if score == "acc":
            tscores.append(Model.score(Xtest, ytest))
        elif score == 'f1':
            ypred = Model.predict(Xtest)
            tscores.append(f1_score(ytest, ypred))
    return tscores
    
def get_average_score(tscores):
    """
    Gets the average score for models cross validated using the cross_validate function.
    :param tscores: List. The scores to be averaged. Should be the result of a cross validation using the cross_validate function.
    """
    score = 0
    for i in range(0,len(tscores)):
        score += tscores[i-1]
    score = score/len(tscores)
    return score
    
    