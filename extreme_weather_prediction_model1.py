import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,classification_report
from sklearn.model_selection import GridSearchCV, cross_validate,KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

pd.set_option("display.max_columns",None)
pd.set_option("display.width",None)
pd.set_option("display.max_rows",20)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv("Datasets/ekstremhavaolayları.csv")
df.head()
#print(df)

def kontrol_df(dataframe,head=5):
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
    numeric_df = dataframe.select_dtypes(include=['number'])
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
#print(kontrol_df(df))

def yakala_sutun_tipini(dataframe, cat_th=10,car_th=20):
    #Katagorik ve kardinaller sütunlar
    kategorik_sutunlar=[col for col in dataframe.columns if dataframe[col].dtypes=="O"]
    numerik_ama_kategorik=[ col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes!="0"]
    kategorik_ama_kardinal=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes=="O"]
    kategorik_sutunlar=kategorik_sutunlar+numerik_ama_kategorik
    kategorik_sutunlar=[col for col in kategorik_sutunlar if col not in kategorik_ama_kardinal]

    #Numerik sütunlar
    numerik_sutunlar=[col for col in dataframe.columns if dataframe[col].dtypes!="O"]
    numerik_sutunlar=[col for col in numerik_sutunlar if col not in numerik_ama_kategorik]

    print(f"Observations:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f"Kategorik Sütunlar:{len(kategorik_sutunlar)}")
    print(f"Numerik Sütunlar:{len(numerik_sutunlar)}")
    print(f'Kategorik ama Kardinal: {len(kategorik_ama_kardinal)}')
    print(f'Numerik ama Kategorik: {len(numerik_ama_kategorik)}')

    return kategorik_sutunlar, numerik_sutunlar, kategorik_ama_kardinal
print(yakala_sutun_tipini(df))

kategorik_sutun,numerik_sutun,kategorik_ama_kardinal=yakala_sutun_tipini(df)

print("#######################################################")

le=LabelEncoder()
for col in kategorik_sutun:
    df[col]=le.fit_transform(df[col])
for col in numerik_sutun:
    df[col] = le.fit_transform(df[col])
for col in kategorik_ama_kardinal:
    df[col] = le.fit_transform(df[col])


f,ax=plt.subplots(figsize=(18,13))
sns.heatmap(df.corr(),annot=True,fmt=".2f",ax=ax,cmap="magma")
ax.set_title("Correlation Matrix",fontsize=20)
#plt.show()

def outlier_thresholds(dataframe,kolon_ismi,q1=0.2,q3=0.8):
    quartile1=dataframe[kolon_ismi].quantile(q1)
    quartile3 = dataframe[kolon_ismi].quantile(q3)
    interquantile_range=quartile3-quartile1
    uplimit=quartile3+1.5*interquantile_range
    lowlimit=quartile3-1.5*interquantile_range
    return lowlimit,uplimit
def check_outlier(dataframe,kolon_ismi):
    lowlimit,uplimit=outlier_thresholds(dataframe,kolon_ismi)
    if dataframe[(dataframe[kolon_ismi]>uplimit)|(dataframe[kolon_ismi]<lowlimit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe,variable,q1=0.1,q3=0.9):
    lowlimit,uplimit=outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable]<lowlimit),variable]=lowlimit
    dataframe.loc[(dataframe[variable]>uplimit),variable]=uplimit

#for col in df.columns:
    print(col,check_outlier(df,col))
    if check_outlier(df,col):
        replace_with_thresholds(df,col)

#for col in df.columns:
    print(col,check_outlier(df,col))

zero_columns=[col for col in df.columns if(df[col].min()==0and col not in ["Attrition"])]
print(zero_columns)

for col in zero_columns:
    df[col]=np.where(df[col]==0,np.nan,df[col])
print(df.isnull().sum())

def eksik_degerler_tablosu(dataframe,na_name=False):
    na_sutunlar=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_eksikler=dataframe[na_sutunlar].isnull().sum().sort_values(ascending=False)
    oran=(dataframe[na_sutunlar].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    kayip_df=pd.concat([n_eksikler, np.round(oran,2)],axis=1,keys=["n_eksikler","oran"])
    print(kayip_df.T,end="\n")
    if na_name:
        return na_sutunlar

na_sutunlar=eksik_degerler_tablosu(df,na_name=True)

for col in df.columns:  # Sadece mevcut sütunlar için işlemi uygula
    if df[col].isnull().sum() > 0:
        df.loc[df[col].isnull(), col] = df[col].median()
#print(df.isnull().sum())

kategorik_sutun,numerik_sutun,kategorik_ama_kardinal=yakala_sutun_tipini(df)

def label_encoder(dataframe,binary_col):
    labelencoder=LabelEncoder()
    dataframe[binary_col]=labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols=[col for col in df.columns if df[col].dtypes =="O" and df[col].nunique()==2]

for col in binary_cols:
    df=label_encoder(df,col)

#One-Hot Encoding işlemi
#Kategorik Sütünların Güncellenmesi
kategorik_sutun=[col for col in kategorik_sutun if col not in binary_cols and col not in["Attrition"]]
print(kategorik_sutun)
def one_hot_encoder(dataframe,categorical_cols, drop_first=False):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
df=one_hot_encoder(df,kategorik_sutun,drop_first=True)

#Standartlaştırma
scaler=StandardScaler()
df[numerik_sutun]=scaler.fit_transform(df[numerik_sutun])
rf_model=RandomForestClassifier(random_state=10)
print(df.head())
print(df.shape)
pipeline = Pipeline([
    ('scaler', scaler),
    ('classifier', rf_model)
])


#Anamoli Algoritması
features = ["AverageTemperature", "AverageTemperatureUncertainty"]


X = df[features]

iso_forest = IsolationForest(contamination=0.05, random_state=10)
df["anomaly"] = iso_forest.fit_predict(X)

# IsolationForest çıktısında:[-1 -> anomaly (anormal durum)/1 ->normal durum]
df["anomaly_flag"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

print("Toplam Anomali Sayısı:", df["anomaly_flag"].sum())



# Hedef değişken: anomali tespiti (1: anomaly, 0: normal)
y = df["anomaly_flag"]
# Özellikler: Anomali dışında kalan sütunlar (örn. AverageTemperature, AverageTemperatureUncertainty, vs.)
# Burada anomali ile ilgili etiket sütunlarını (anomaly, anomaly_flag) çıkarıyoruz.
X = df.drop(["anomaly", "anomaly_flag"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=10)


cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

print(f"Cross Validation F1 Scores: {cv_scores}")
print(f"Ortalama F1 Score: {cv_scores.mean()}")



param_grid = {
    'classifier__n_estimators': [20, 50,100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)


best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"AUC: {round(roc_auc_score(y_pred, y_test), 2)}")








