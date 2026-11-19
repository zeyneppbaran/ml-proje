"""Ortak Pipeline ve ColumnTransformer Bilesenleri.

Bu dosya, on isleme bilesenlerini ve siniflandiricilari uctan uca
birlestiren standart Scikit-Learn ve imbalanced-learn boru hatlarini (Pipeline) sunar.
"""
from sklearn.pipeline import Pipeline
from .preprocessing import create_preprocessor

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


def get_preprocessor():
    """Tum modeller icin standart ColumnTransformer on isleyicisini dondurur."""
    return create_preprocessor()


def build_full_pipeline(classifier):
    """
    Standart bir siniflandiriciyi ortak on isleme bileseniyle tek bir Pipeline altinda birlestirir.
    Veri sizintisini (data leakage) onler; fit islemi sadece egitim verisinde yapilir.
    """
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", classifier)
    ])


def build_smote_pipeline(classifier, random_state=42):
    """
    SMOTE dengeleme adimini ve siniflandiriciyi imblearn Pipeline ile guvenli birlestirir.
    SMOTE sadece egitim katlarinda uygulanir; test ve dogrulama verisine asla dokunulmaz.
    """
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imblearn kütüphanesi yüklü değil. 'pip install imbalanced-learn' çalıştırınız.")

    return ImbPipeline([
        ("preprocessor", get_preprocessor()),
        ("smote", SMOTE(random_state=random_state)),
        ("classifier", classifier)
    ])
