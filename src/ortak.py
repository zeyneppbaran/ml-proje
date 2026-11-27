"""Tum notebook'larin kullandigi ortak fonksiyonlar.

Bu dosyayi sadece Kisi 3 degistirir. Degisiklik isteyen ona soyler.
Amac: Bes modelin de ayni veriyi, ayni bolmeyi ve ayni on islemeyi kullanmasi.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# Ortak on isleme bilesenleri
from .preprocessing import create_preprocessor, prepare_unknown_values

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
TEMP_SIZE = 0.30
CV_SPLITS = 5
HEDEF_SUTUN = "y"
VERI_YOLU = "data/bank-additional-full.csv"
ORTAK_SKORLAR = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _bul_veri_yolu():
    """Veri setinin konumunu farkli calisma dizinlerinde dinamik olarak tespit eder."""
    aday_yollar = [
        Path(VERI_YOLU),
        Path("../" + VERI_YOLU),
        Path(__file__).resolve().parent.parent / VERI_YOLU
    ]
    for p in aday_yollar:
        if p.exists():
            return str(p)
    return VERI_YOLU


def veri_yukle(veri_yolu=None):
    """Ham veriyi yukler, duration henuz varken gercek tekrarlari siler ve ortak donusumleri uygular.

    Hedef sutun yes=1, no=0 olarak kodlanir.
    """
    if veri_yolu is None:
        veri_yolu = _bul_veri_yolu()

    if not os.path.exists(veri_yolu):
        raise FileNotFoundError(f"Veri seti bulunamadi: {veri_yolu}")

    # Veri setini noktali virgul ayiricisiyla oku ve duration henuz mevcutken gercek tekrarlari sil
    df = pd.read_csv(veri_yolu, sep=';')
    df = df.drop_duplicates().reset_index(drop=True).copy()

    # Dusuk oranli kategorik sutunlarda 'unknown' -> NaN (default korunur)
    df = prepare_unknown_values(df)

    # Daha once aranma bilgisi (onceki kampanyalarda en az 1 temas yapilmis mi?)
    df["daha_once_aranmis"] = (df["previous"] > 0).astype(int)

    # Son aramadan bu yana gecen gun bilgisinin kayitli olup olmadigi (pdays != 999)
    df["pdays_kayitli"] = (df["pdays"] != 999).astype(int)

    # 999 gercek bir gun sayisi degildir; NaN yapilir ve ortak Pipeline icindeki
    # SimpleImputer(strategy='median') yalnizca egitim katindan ogrenerek doldurur.
    df["pdays"] = df["pdays"].replace(999, np.nan)

    # Hedef sutun yes=1, no=0 olarak kodlanir
    df[HEDEF_SUTUN] = df[HEDEF_SUTUN].map({"yes": 1, "no": 0})
    return df


def veriyi_bol(df):
    """
    Veriyi her zaman ayni sekilde %70 Egitim, %15 Dogrulama (Validation) ve %15 Test olarak boler.

    ONEMLI (VERI SIZINTISI / DATA LEAKAGE ONLEMI):
    - 'duration' (gorusme suresi) sutunu modelden kesinlikle cikarilmistir.
      Gerekce: Bir arama gerceklesmeden once telefon gorusmesinin kac saniye surecegi bilinemez.
      Arama bittiginde ise musterinin vadeli mevduat acip acmadigi zaten bellidir.
    - On isleme (imputer, scaler, encoder) ve SMOTE yalnizca X_train uzerinden ogrenilmelidir.
    - X_val model secimi ve hiperparametre dogrulamasinda, X_test yalnizca nihai degerlendirmede kullanilir.

    Donus:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[HEDEF_SUTUN, "duration"])
    y = df[HEDEF_SUTUN]

    # 1. Asama: %70 Egitim, %30 Gecici (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEMP_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 2. Asama: Kalan %30'u %15 Validation ve %15 Test olarak esit bol
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def on_isleyici():
    """Ortak ColumnTransformer'i dondurur."""
    return create_preprocessor()


def ortak_cv(n_splits=CV_SPLITS, random_state=RANDOM_STATE):
    """
    Tum modellerin ortak kullanacagi 5 katli tabakali capraz dogrulama (StratifiedKFold) nesnesini dondurur.

    Kurallar:
    - n_splits=5, shuffle=True, random_state=42 sabitleri tum modellerde aynidir.
    - Yalnizca egitim verisi (X_train, y_train) uzerinde kullanilir; validation ve test setleri dahil edilmez.
    - Pipeline ile birlikte kullanildiginda on isleme her fold icinde yalnizca ilgili egitim fold'unda fit edilir.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def capraz_dogrula(pipeline, X_train, y_train, cv=None, scoring=None, n_jobs=-1):
    """
    Verilen ortak Pipeline'i yalnizca egitim verisi uzerinde 5 katli capraz dogrulama ile degerlendirir.

    Validation (X_val, y_val) ve test (X_test, y_test) setleri bu fonksiyona verilmemelidir.
    Her fold icinde on isleme (ve varsa SMOTE) yalnizca ilgili egitim katinda fit edilir.

    Parametreler:
        pipeline: Scikit-learn veya imblearn Pipeline nesnesi.
        X_train: Egitim ozellik matrisi.
        y_train: Egitim hedef degiskeni (0/1).
        cv: Opsiyonel capraz dogrulama bolucusu (varsayilan: ortak_cv()).
        scoring: Opsiyonel skor sozlugu (varsayilan: ORTAK_SKORLAR).
        n_jobs: Paralel islem sayisi (varsayilan: -1).

    Donus:
        dict: Her metrik icin 'ortalama', 'std' ve 'kat_skorlari' degerlerini iceren sozluk.
    """
    if cv is None:
        cv = ortak_cv()
    if scoring is None:
        scoring = ORTAK_SKORLAR

    cv_sonuc = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    ozet = {}
    for metrik in scoring:
        skorlar = cv_sonuc[f"test_{metrik}"]
        ozet[metrik] = {
            "ortalama": float(np.mean(skorlar)),
            "std": float(np.std(skorlar)),
            "kat_skorlari": [float(s) for s in skorlar],
        }
    return ozet


def degerlendir(model, X_test, y_test, model_adi, klasor="results"):
    """
    Test metriklerini hesaplar, ekrana yazdirir ve results/ klasorune json olarak kaydeder.
    y_test ve tahminler 0/1 olarak kodlanmis olmalidir (1 = pozitif sinif).
    """
    y_pred = model.predict(X_test)
    sonuc = {
        "model": model_adi,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        sonuc["roc_auc"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    elif hasattr(model, "decision_function"):
        sonuc["roc_auc"] = float(roc_auc_score(y_test, model.decision_function(X_test)))

    os.makedirs(klasor, exist_ok=True)
    with open(os.path.join(klasor, model_adi + ".json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=2, ensure_ascii=False)

    print(f"--- {model_adi} Test Performansı ---")
    for k, v in sonuc.items():
        if isinstance(v, float):
            print(f"{k:12s}: {v:.4f}")
        else:
            print(f"{k:12s}: {v}")
    return sonuc


def sonuclari_oku(klasor="results"):
    """results/ klasorundeki tum json dosyalarini tek tabloda birlestirir."""
    satirlar = []
    if not os.path.exists(klasor):
        return pd.DataFrame()
    for dosya in sorted(os.listdir(klasor)):
        if dosya.endswith(".json"):
            with open(os.path.join(klasor, dosya), encoding="utf-8") as f:
                satirlar.append(json.load(f))
    return pd.DataFrame(satirlar)
