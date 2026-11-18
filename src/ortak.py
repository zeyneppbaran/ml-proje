"""Tum notebook'larin kullandigi ortak fonksiyonlar.

Bu dosyayi sadece Kisi 3 degistirir. Degisiklik isteyen ona soyler.
Amac: bes modelin de ayni veriyi, ayni bolmeyi ve ayni on islemeyi kullanmasi.
"""
import json
import os

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.2
HEDEF_SUTUN = "y" 


VERI_YOLU = "data/bank-additional-full.csv"

def veri_yukle():
    """Ham veriyi yukler. Hedef sutun yes=1, no=0 olarak kodlanir."""

    if os.path.exists(VERI_YOLU):
        df = pd.read_csv(VERI_YOLU, sep=';')
    else:
        raise FileNotFoundError(
            f"Veri seti bulunamadi: {VERI_YOLU}"
        )

    df[HEDEF_SUTUN] = df[HEDEF_SUTUN].map({
        "yes": 1,
        "no": 0
    })

    return df
def veriyi_bol(df):
    """Veriyi her zaman ayni sekilde egitim ve test olarak boler."""
    X = df.drop(columns=[HEDEF_SUTUN,"duration"])#duration sutunu hedefi etkiledigi icin cikarildi
    y = df[HEDEF_SUTUN]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def on_isleyici():
    """Ortak ColumnTransformer'i dondurur (eksik doldurma, kodlama, olcekleme)."""
    # TODO: Kisi 3, notebook 03'te gelistirip buraya tasiyacak
    raise NotImplementedError("on_isleyici() henuz yazilmadi")


def degerlendir(model, X_test, y_test, model_adi, klasor="results"):
    """Test metriklerini hesaplar, ekrana yazar ve results/ klasorune kaydeder.

    y_test ve tahminler 0/1 olarak kodlanmis olmalidir (1 = pozitif sinif).
    """
    y_pred = model.predict(X_test)
    sonuc = {
        "model": model_adi,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    if hasattr(model, "predict_proba"):
        sonuc["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    elif hasattr(model, "decision_function"):
        sonuc["roc_auc"] = roc_auc_score(y_test, model.decision_function(X_test))
    os.makedirs(klasor, exist_ok=True)
    with open(os.path.join(klasor, model_adi + ".json"), "w") as f:
        json.dump(sonuc, f, indent=2)
    for k, v in sonuc.items():
        print(k, ":", v if isinstance(v, str) else round(v, 4))
    return sonuc


def sonuclari_oku(klasor="results"):
    """results/ klasorundeki tum json dosyalarini tek tabloda birlestirir."""
    satirlar = []
    for dosya in sorted(os.listdir(klasor)):
        if dosya.endswith(".json"):
            with open(os.path.join(klasor, dosya)) as f:
                satirlar.append(json.load(f))
    return pd.DataFrame(satirlar)
