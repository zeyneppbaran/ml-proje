
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sayısal sütunlar
numeric_features = [
    'age', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

# En sık görülen değerle doldurulacak kategorik sütunlar
mode_features = [
    'job', 'marital', 'education', 'housing', 'loan'
]

# unknown değerini ayrı kategori olarak koruyacağımız sütun
unknown_features = ['default']

# Diğer kategorik sütunlar
other_cat_features = [
    'contact', 'month', 'day_of_week', 'poutcome'
]


def prepare_unknown_values(df):
    """
    Seçilen sütunlardaki 'unknown' değerlerini eksik değere dönüştürür.
    Bu işlem veri öğrenmez; sadece metni NaN yapar.
    """
    df = df.copy()
    df[mode_features] = df[mode_features].replace('unknown', np.nan)
    return df


def create_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    mode_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    unknown_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    other_cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('mode', mode_transformer, mode_features),
            ('unknown', unknown_transformer, unknown_features),
            ('other_cat', other_cat_transformer, other_cat_features)
        ]
    )

    return preprocessor
