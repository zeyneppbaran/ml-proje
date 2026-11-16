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

RANDOM_STATE = 42
TEST_SIZE = 0.2
HEDEF_SUTUN = "HEDEF_SUTUN_ADI"  # TODO: veri seti secilince doldurulacak


def veri_yukle():
    """Veri setini okuyup DataFrame olarak dondurur."""
    # TODO: veri seti secilince doldurulacak (ornek: pd.read_csv(URL))
    raise NotImplementedError("veri_yukle() henuz yazilmadi")


def veriyi_bol(df):
    """Veriyi her zaman ayni sekilde egitim ve test olarak boler."""
    X = df.drop(columns=[HEDEF_SUTUN])
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
