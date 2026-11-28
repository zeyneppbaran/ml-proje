"""Bes Modelin Ortak Pipeline ve SMOTE Pipeline Uyumluluk Testi (Kisi 3).

Bu test dosyasi asagidaki teknik uyumluluk kontrollerini gerceklestirir:
1. Ortak veri_yukle() ve veriyi_bol() (%70 Egitim / %15 Validation / %15 Test) kullanimi.
2. Bes modelin (Logistic Regression, KNN, SVM, Random Forest, XGBoost)
   build_full_pipeline() ile:
   - Pipeline olusturma,
   - Kucuk egitim alt kumesinde (X_train icinden) fit(),
   - Ayri dogrulama alt kumesinde (X_val icinden) predict(),
   - Tahmin boyutu ve gecerli sinif degerleri ({0, 1}) dogrulamasi.
3. Ayni bes modelin build_smote_pipeline() ile ayri teknik uyumluluk ve
   SMOTE'un yalnizca egitim asamasinda calistiginin dogrulamasi.

Not: Bu dosya yalnizca teknik uyumluluk testidir; gercek model performansi
veya modeller arasi basari karsilastirmasi hesaplamaz. Test verisi (X_test, y_test)
kesinlikle kullanilmaz.

Calistirma:
    pytest tests/test_bes_model_pipeline.py -v -s
"""
import sys
from pathlib import Path

PROJE_KOKU = Path(__file__).resolve().parent.parent
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from src.ortak import RANDOM_STATE, veri_yukle, veriyi_bol
from src.pipeline import build_full_pipeline, build_smote_pipeline


def _model_olustur(model_adi):
    """Test edilecek 5 siniflandiriciyi varsayilan/hafif test ayarlariyla dondurur."""
    modeller = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "SVM": lambda: SVC(random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }
    return modeller[model_adi]()


MODEL_ADLARI = [
    "Logistic Regression",
    "KNN",
    "SVM",
    "Random Forest",
    "XGBoost",
]


@pytest.fixture(scope="module")
def kucuk_egitim_ve_dogrulama_verisi():
    """
    Ortak veri_yukle() ve veriyi_bol() (%70/%15/%15) fonksiyonlarini kullanarak:
    - Yalnizca X_train icinden 500 orneklik tabakali egitim alt kumesi,
    - Yalnizca X_val icinden 100 orneklik ayri tabakali dogrulama alt kumesi olusturur.
    Test seti (X_test, y_test) kesinlikle kullanilmaz.
    """
    df = veri_yukle()
    X_train, X_val, X_test, y_train, y_val, y_test = veriyi_bol(df)

    # Veri sizintisi guvencesi: uc kumenin indeksleri tamamen ayrik ve duration yok
    assert "duration" not in X_train.columns
    assert "duration" not in X_val.columns
    assert "duration" not in X_test.columns
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    X_tr_kucuk, _, y_tr_kucuk, _ = train_test_split(
        X_train,
        y_train,
        train_size=500,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    X_va_kucuk, _, y_va_kucuk, _ = train_test_split(
        X_val,
        y_val,
        train_size=100,
        stratify=y_val,
        random_state=RANDOM_STATE,
    )

    assert set(X_tr_kucuk.index).isdisjoint(X_va_kucuk.index)
    return X_tr_kucuk, y_tr_kucuk, X_va_kucuk, y_va_kucuk


@pytest.mark.parametrize("model_adi", MODEL_ADLARI)
def test_ortak_full_pipeline_uyumlulugu(model_adi, kucuk_egitim_ve_dogrulama_verisi):
    """Her modelin build_full_pipeline() ile olusturma, fit ve predict adimlarini dogrular."""
    X_tr, y_tr, X_va, _ = kucuk_egitim_ve_dogrulama_verisi

    clf = _model_olustur(model_adi)
    pipe = build_full_pipeline(clf)

    # 1. Pipeline basariyla olusturuldu mu?
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps

    # 2. Yalnizca egitim alt kumesiyle fit()
    pipe.fit(X_tr, y_tr)

    # 3. Ayri dogrulama alt kumesiyle predict()
    tahminler = pipe.predict(X_va)

    # 4. Tahmin boyutu ve gecerli sinif etiketleri (0 ve 1)
    assert tahminler.shape == (len(X_va),), (
        f"{model_adi} beklenen boyut {(len(X_va),)}, alinan {tahminler.shape}"
    )
    assert not np.isnan(tahminler).any(), f"{model_adi} tahminlerinde NaN bulundu!"
    assert set(np.unique(tahminler)).issubset({0, 1}), (
        f"{model_adi} gecersiz sinif etiketleri uretti: {np.unique(tahminler)}"
    )
    print(f"[BASARILI] Full Pipeline | {model_adi:20s} | fit=(500,) | predict={tahminler.shape}")


@pytest.mark.parametrize("model_adi", MODEL_ADLARI)
def test_smote_pipeline_uyumlulugu(model_adi, kucuk_egitim_ve_dogrulama_verisi):
    """Her modelin build_smote_pipeline() ile egitim-ici SMOTE, fit ve predict adimlarini dogrular."""
    X_tr, y_tr, X_va, _ = kucuk_egitim_ve_dogrulama_verisi

    clf = _model_olustur(model_adi)
    smote_pipe = build_smote_pipeline(clf, random_state=RANDOM_STATE)

    # 1. imblearn Pipeline ve adim sirasi: preprocessor -> smote -> classifier
    assert isinstance(smote_pipe, ImbPipeline)
    assert list(smote_pipe.named_steps.keys()) == ["preprocessor", "smote", "classifier"]

    # 2. Yalnizca egitim alt kumesiyle fit() (SMOTE yalnizca burada calisir)
    smote_pipe.fit(X_tr, y_tr)

    # 3. Ayri dogrulama alt kumesiyle predict() (SMOTE dogrulama verisine uygulanmaz)
    tahminler = smote_pipe.predict(X_va)

    # 4. Tahmin boyutu X_va ile birebir ayni ve gecerli sinif degerleri ({0, 1})
    assert tahminler.shape == (len(X_va),), (
        f"SMOTE + {model_adi} beklenen boyut {(len(X_va),)}, alinan {tahminler.shape}"
    )
    assert not np.isnan(tahminler).any(), f"SMOTE + {model_adi} tahminlerinde NaN bulundu!"
    assert set(np.unique(tahminler)).issubset({0, 1}), (
        f"SMOTE + {model_adi} gecersiz sinif etiketleri uretti: {np.unique(tahminler)}"
    )
    print(f"[BASARILI] SMOTE Pipeline | {model_adi:20s} | fit=(500->dengeli) | predict={tahminler.shape}")


def test_smote_dogrulama_verisini_degistirmez(kucuk_egitim_ve_dogrulama_verisi):
    """SMOTE'un egitim siniflarini dengeledigini, dogrulama verisini ise hic degistirmedigini dogrular."""
    X_tr, y_tr, X_va, y_va = kucuk_egitim_ve_dogrulama_verisi
    y_va_orijinal = y_va.copy()

    smote_pipe = build_smote_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    smote_pipe.fit(X_tr, y_tr)

    # Egitim verisi preprocessor sonrasi SMOTE ile esitlenmis olmali
    X_tr_islenmis = smote_pipe.named_steps["preprocessor"].transform(X_tr)
    _, y_tr_dengeli = smote_pipe.named_steps["smote"].fit_resample(X_tr_islenmis, y_tr)
    assert int((y_tr_dengeli == 0).sum()) == int((y_tr_dengeli == 1).sum())
    assert len(y_tr_dengeli) > len(y_tr)

    # Dogrulama verisi predict sirasinda cogaltilmamis ve bozulmamis olmali
    tahminler = smote_pipe.predict(X_va)
    assert len(tahminler) == len(X_va) == len(y_va_orijinal)
    assert (y_va.values == y_va_orijinal.values).all()
    print(
        f"[BASARILI] SMOTE sizinti kontrolu: egitim {len(y_tr)} -> {len(y_tr_dengeli)} (dengeli), "
        f"dogrulama sabit ({len(X_va)})."
    )
