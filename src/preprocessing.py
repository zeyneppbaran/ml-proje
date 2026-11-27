"""Bank Marketing Veri Seti Ortak On Isleme Modulu.

Bu modul, tum ekip uyelerinin (5 model sahibi) ortak kullanacagi on isleme
adimlari olan ColumnTransformer mimarisini ve unknown/eksik deger donusumlerini icerir.
Veri sizintisi (data leakage) olusmamasi icin fit islemi sadece egitim verisinde yapilmalidir.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sayisal sutunlar (pdays ozel kodu ve yeni turetilen daha_once_aranmis, pdays_kayitli dahil)
numeric_features = [
    'age', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed',
    'daha_once_aranmis',
    'pdays_kayitli'
]

# En sik gorulen degerle (mode / most_frequent) doldurulacak kategorik sutunlar (Eksik orani < %5)
mode_features = [
    'job', 'marital', 'education', 'housing', 'loan'
]

# 'unknown' degeri bilgi tasidigi icin ayri bir kategori olarak korunacak sutun (Eksik orani %20.87)
unknown_features = ['default']

# Diger kategorik sutunlar (Dogrudan One-Hot Encoding uygulanacaklar)
other_cat_features = [
    'contact', 'month', 'day_of_week', 'poutcome'
]


def prepare_unknown_values(df):
    """
    Secilen dusuk oranli sutunlardaki 'unknown' degerlerini eksik degere (NaN) donusturur.
    Bu islem veri ogrenmez; sadece metni NaN yapar.
    'default' sutunundaki unknown degeri finansal risk belirsizligi tasidigi icin korunur.
    """
    df = df.copy()
    df[mode_features] = df[mode_features].replace('unknown', np.nan)
    return df


def create_preprocessor():
    """
    Tum modellerin kullanacagi ortak scikit-learn ColumnTransformer nesnesini olusturur.
    
    Adimlar:
    - Sayisal sutunlar: SimpleImputer(median) + StandardScaler
    - Mode sutunlar: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
    - Default sutunu: OneHotEncoder(handle_unknown='ignore')
    - Diger kategorikler: OneHotEncoder(handle_unknown='ignore')
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    mode_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    unknown_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    other_cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('mode', mode_transformer, mode_features),
            ('unknown', unknown_transformer, unknown_features),
            ('other_cat', other_cat_transformer, other_cat_features)
        ],
        remainder='drop'
    )

    return preprocessor
