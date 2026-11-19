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
from sklearn.model_selection import train_test_split

# ortak.py'nin import kismina ekle:
from .preprocessing import create_preprocessor, prepare_unknown_values

RANDOM_STATE = 42
TEST_SIZE = 0.2
HEDEF_SUTUN = "y"
VERI_YOLU = "data/bank-additional-full.csv"


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
    """Ham veriyi yukler. Hedef sutun yes=1, no=0 olarak kodlanir."""
    if veri_yolu is None:
        veri_yolu = _bul_veri_yolu()

    if not os.path.exists(veri_yolu):
        raise FileNotFoundError(f"Veri seti bulunamadi: {veri_yolu}")

    # Veri setini noktali virgul ayiricisiyla oku
    df = pd.read_csv(veri_yolu, sep=';')

    # 'unknown' -> NaN (eklendi)
    df = prepare_unknown_values(df)

    # pdays ozel kodu cozumleme: 999 degeri sayisal bir buyukluk degil, ozel koddur!
    df["daha_once_aranmis"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)

    # Hedef sutun yes=1, no=0 olarak kodlanir
    df[HEDEF_SUTUN] = df[HEDEF_SUTUN].map({"yes": 1, "no": 0})
    return df


def veriyi_bol(df):
    """
    Veriyi her zaman ayni sekilde egitim ve test olarak boler.
    
    ONEMLI (VERI SIZINTISI / DATA LEAKAGE ONLEMI):
    'duration' (gorusme suresi) sutunu modelden kesinlikle cikarilmistir!
    Gerekce: Bir arama gerceklesmeden once telefon gorusmesinin kac saniye surecegi bilinemez.
    Arama bittiginde ise musterinin vadeli mevduat acip acmadigi zaten bellidir.
    Bu sutunun modele verilmesi yapay olarak yuksek basari uretir ancak sahada calismaz;
    yani dogrudan bir Veri Sizintisi (Data Leakage) olusturur.
    """
    X = df.drop(columns=[HEDEF_SUTUN, "duration"])
    y = df[HEDEF_SUTUN]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def on_isleyici():
    """Ortak ColumnTransformer'i dondurur."""
    return create_preprocessor()


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
