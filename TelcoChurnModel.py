#Kütüphane tanımla

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#Veri oku

df = pd.read_csv("W6/HW2/Telco-Customer-Churn.csv")
df.head()


#Görev1

#****************************************Adım1 - Genel Resmi inceleyiniz.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

#Yorum:customerID silinebilir. TotalCharges int olmalı.Sadece totalCharges'ta 11 boş değer var.
#Target değer object. İlerideki analizler için numeric yapmalıyız.

df["Churn"]= df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors = 'coerce')
df.drop("customerID",axis=1,inplace=True)
df.head()
#TotalCharges daki güncellemeden sonra TotalCharges için boş olan 11 satır oluştu.
#*********************************Adım2 - Kategorik ve numeric değişkenleri yakalayınız

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Yorum: 17 kategorik,3 numerik kolon var

#*********************************Adım3-Numerik ve kategorik değişkenlerin analizini yapınız

#Kategorik Değişken Analizi

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

for col in cat_cols:
    cat_summary(df, col)

#Yorum: Dependents,PhoneService,SeniorCitizen oran farkları yüksek
#No internet service ve no phone service olan kolonlar sadece no yapılabilir.
#Bağımlı değişken oranları %27-%73. Data unbalanced

#Numeric değişken analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


#Yorum tenure sıfır olanların monthly Charges değeri boş. Çünkü henüz 1 ayları dolmamış
#Boş değerler silinebilir , 1 aylık değerleri yazdırılabilir veya sıfır yazılabilir.
#Total ve monthly charge qcut ile segmente edilebilir.

df[df["TotalCharges"].isnull()]

#**********************************************Adım4 - Hedef değişken analizi yapınız.
# Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

#Sayısal Değişkenler ile target value

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, 'Churn', col)


###########################################
#Kategorik Değişkenler ile target value
############################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, 'Churn', col)

######################################################################################
#########################Hiçbir değişiklik yapmdan model kur############################
######################################################################################

df_copy=df.copy()
df_copy.dropna(inplace=True)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in ["Churn"]]
dff = one_hot_encoder(df_copy, cat_cols, drop_first=True)
dff.head()

y = dff["Churn"]
X = dff.drop(["Churn"], axis=1)

rf_model = RandomForestClassifier(random_state=17).fit(X,y)
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
#Acc:79,F1:55,Auc:82

#Feature importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X, 5)

#Önem dereceleri sırayla => TotalCharges,tenure,monthly charges,PaymentMethod_ElectronicCheck,InternetService_FiberOptic

#*******************************************Adım5 - Aykırı Gözlem analizi yapınız.

#1) upper ve lower bound bul- 1-99 ve 5-95  aralıklarını deneyip karşılaştıracağım.

def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    print(col, outlier_thresholds(df, col))


#2) Elde edilen sınırlara göre aykırı değer var mı?

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

#Herhangi Outlier gözükmüyor.

###############################################################################
#*********************************************Adım6 - Eksik Gözlem Analizi Yapınız
###############################################################################

df.isnull().sum()

#Yorum: tenure değerleri 0 olan 11 satırda TotalCharges boş. Bu satırlar silinebilir, 0 kabul edilebilir veya MonthlyCharges kopyalanabilir.
#Ben monthly charge kopyalayarak devam ettim.
df.head()

df.loc[(df["TotalCharges"].isnull()) , "TotalCharges"] =df[df["TotalCharges"].isnull()]["MonthlyCharges"]
df.isnull().sum()

#**********************************************Adım7: Korelasyon Analizi Yapınız

sns.set(rc={'figure.figsize': (50, 12)})
sns.heatmap(df.corr(), cmap="RdBu",annot=True)
plt.show()

#Yorum - tenure ile total charge arasında yüksek korelasyon var.Fakat %95 ve üzeri bir korelasyon şu an için yok.


#**************************Görev2 - Feature Engineering***********************

df.head(20)

df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

"""#No phone service ve no internet servisi no yapmak
for col in df.columns:
    df.loc[df[col].isin(["No phone service"]), col] = "No"

for col in df.columns:
    df.loc[df[col].isin(["No internet service"]), col] = "No"""

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

df.loc[(df['Contract'] == "Month-to-month" ), "NEW_CONTRACT"] = 1
df.loc[(df['Contract'] == "One year" ), "NEW_CONTRACT"] = 12
df.loc[(df['Contract'] == "Two year" ), "NEW_CONTRACT"] = 24

df.info()
df.head()

#########################################################################
######################Kolonları tekrar incele############################

#tenure ve contract'ı çıkaralım.
df2=df.copy()
df2.head()
df2.drop(['tenure','Contract'],axis=1,inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df2)


# NEW_TotalServices ve NEW_CONTRACT değişkeni cat_cols arasında yer almış fakat numeric bir değişken onun yerini değiştirelim.
cat_cols.remove("NEW_CONTRACT")
num_cols.append("NEW_CONTRACT")

cat_cols.remove("NEW_TotalServices")
num_cols.append("NEW_TotalServices")

#num_cols.remove("tenure")
#num_cols.remove("Contract")

#Tekrar outlier bakalım.

for col in num_cols:
    print(col, ":", check_outlier(df2, col))

#Tekrar missing value kaldı mı bakalım

df2.isnull().sum()

#Korelasyon bakalım

def high_correlated_cols(dataframe, plot=False, corr_th=0.95):
    num_cols = grab_col_names(df2)[1]
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu",annot=True)
        plt.show()
    return drop_list

high_correlated_cols(df2, plot=True, corr_th=0.95)

#New Avg Charges ile Monthly Charges arasında yüksek korelasyon var.

high_correlated_col_df = high_correlated_cols(df, corr_th=0.95)

##########################################################################
##################################ENCODING################################
########################################################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in ["Churn","Contract"]]
df2_encode = one_hot_encoder(df2, cat_cols, drop_first=True)
df2_encode.head()

#####################################################################
#################BASE MODEL#################################
###################################################################

#Uzaklık bazlı modeller kullanmayacağım için scale etmiyorum.

# Bağımlı - Bağımsız değişkenler:
X = df2_encode.drop("Churn", axis=1)
y = df2_encode["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

#Bağımlı değişken unbalanced idi smote uygulayalım.
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)

def GenerateModel(ModelName,X,y,params,Hyperparameter=False):
    if Hyperparameter==False:
        model=ModelName.fit(X,y)
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
    else:
        rf_best_grid = GridSearchCV(ModelName, params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

        final_model = ModelName.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
        cv_results = cross_validate(final_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

    print("AccuracyScore "+ str(cv_results['test_accuracy'].mean()))
    print("F1 Score " + str(cv_results['test_f1'].mean()))
    print("Auc Score " + str(cv_results['test_roc_auc'].mean()))

#Model1 Random Forest
GenerateModel(RandomForestClassifier(),X_smote,y_smote,False)

##Acc 83, f1 83,  AUC 92 oldu :))))

params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

GenerateModel(RandomForestClassifier(),X_smote,y_smote,True)

#83,83,92

#Model2 GBM

params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

#Hyperparameter olmadan sonuçlar
GenerateModel(GradientBoostingClassifier(),X_smote,y_smote,params,False)
#83,82,91

#Optimizasyon sonrası sonuçlar
GenerateModel(GradientBoostingClassifier(),X_smote,y_smote,params,True)
#84,83,92

#Model3 XGB

params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

#Hyperparameter olmadan sonuçlar
GenerateModel(XGBClassifier(),X_smote,y_smote,params,False)
#83,82,92

#Optimizasyon sonrası sonuçlar
GenerateModel(XGBClassifier(),X_smote,y_smote,params,True)
#84,83,93

#Model4 LightGBM

params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

#Hyperparameter olmadan sonuçlar
GenerateModel(LGBMClassifier(),X_smote,y_smote,params,False)
#83,83,93

#Optimizasyon sonrası sonuçlar
GenerateModel(LGBMClassifier(),X_smote,y_smote,params,True)
#84,83,92

#Bence lightGBM, oldukç hızlı
