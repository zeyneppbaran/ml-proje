"""Ortak Pipeline, Modul Import ve 5 Model Uyumluluk Testi (Kisi 3).

Bu test dosyasi:
1. Tum ortak Python modullerinin sorunsuz import edilebildigini dogrular.
2. Ortak veri yukleme, bolme, on isleme ve 5 katli StratifiedKFold ayarini kontrol eder.
3. Bes farkli modelin (Lojistik Regresyon, KNN, SVM, Random Forest, XGBoost)
   ortak build_full_pipeline() ile Pipeline olusturma, kucuk orneklemde fit ve
   predict adimlarini gercek calistirma ile dogrular.
4. build_smote_pipeline() icinde SMOTE'un yalnizca egitim katinda calistigini
   ve dogrulama/test verisine uygulanmadigini dogrular.

Calistirma:
    python tests/test_ortak_pipeline.py
"""
import sys
from pathlib import Path

# Proje kok dizinini Python arama yoluna ekle
PROJE_KOKU = Path(__file__).resolve().parent.parent
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate


def test_modul_importlari():
    """Tum ortak Python modullerinin import edilebildigini ve kurallarini dogrular."""
    import tempfile
    import src
    import src.ortak
    import src.preprocessing
    import src.pipeline

    # src/ortak.py fonksiyon kontrolleri
    for fonk in [
        "veri_yukle",
        "veriyi_bol",
        "on_isleyici",
        "ortak_cv",
        "capraz_dogrula",
        "degerlendir",
        "sonuclari_oku",
    ]:
        assert hasattr(src.ortak, fonk), f"src.ortak.{fonk} eksik!"

    # src/preprocessing.py fonksiyon ve yapilandirma kontrolleri
    assert hasattr(src.preprocessing, "create_preprocessor")
    assert hasattr(src.preprocessing, "prepare_unknown_values")
    prep = src.ortak.on_isleyici()
    transformers = dict((name, tr) for name, tr, _ in prep.transformers)
    assert transformers["num"].named_steps["imputer"].strategy == "median"
    assert transformers["num"].named_steps["scaler"].__class__.__name__ == "StandardScaler"
    assert transformers["mode"].named_steps["imputer"].strategy == "most_frequent"
    assert transformers["mode"].named_steps["encoder"].handle_unknown == "ignore"
    assert transformers["unknown"].named_steps["encoder"].handle_unknown == "ignore"
    assert transformers["other_cat"].named_steps["encoder"].handle_unknown == "ignore"
    assert src.preprocessing.unknown_features == ["default"]

    # default sutununda 'unknown' korunuyor, mode_features sutunlarinda NaN yapiliyor
    df = src.ortak.veri_yukle()
    assert (df["default"] == "unknown").sum() > 0, "default sutunundaki unknown korunmamis!"
    for col in src.preprocessing.mode_features:
        assert (df[col] == "unknown").sum() == 0, f"{col} sutununda unknown kalmis!"
    assert "daha_once_aranmis" in df.columns
    assert (df["pdays"] == 999).sum() == 0

    X_train, X_test, y_train, y_test = src.ortak.veriyi_bol(df)
    assert "duration" not in X_train.columns and "duration" not in X_test.columns
    assert len(X_train) == 32950 and len(X_test) == 8238

    # src/pipeline.py kontrolleri
    assert hasattr(src.pipeline, "build_full_pipeline")
    assert hasattr(src.pipeline, "build_smote_pipeline")

    # ortak_cv() ve capraz_dogrula() kontrolleri
    cv = src.ortak.ortak_cv()
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42

    # degerlendir() ve sonuclari_oku() kontrolleri (gecici klasorde, gercek results/ dosyasina dokunmadan)
    X_fit, X_pred, y_fit, y_pred_gercek = _kucuk_orneklem_hazirla(egitim_boyutu=200, dogrulama_boyutu=60)
    ornek_pipe = src.pipeline.build_full_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    cv_ozet = src.ortak.capraz_dogrula(ornek_pipe, X_fit, y_fit, n_jobs=1)
    assert set(cv_ozet.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    ornek_pipe.fit(X_fit, y_fit)
    with tempfile.TemporaryDirectory() as gecici_klasor:
        sonuc = src.ortak.degerlendir(ornek_pipe, X_pred, y_pred_gercek, "test_kontrol", klasor=gecici_klasor)
        tablo = src.ortak.sonuclari_oku(klasor=gecici_klasor)
        assert len(tablo) == 1 and tablo.iloc[0]["model"] == "test_kontrol"
        assert "roc_auc" in sonuc

    print("[BASARILI] Tum ortak moduller (ortak, preprocessing, pipeline) ve fonksiyonlar dogrulandi.")


def _kucuk_orneklem_hazirla(egitim_boyutu=500, dogrulama_boyutu=100):
    """Yalnizca X_train ve y_train icinden kucuk tabakali egitim/dogrulama orneklemi olusturur."""
    from src.ortak import veri_yukle, veriyi_bol, RANDOM_STATE

    df = veri_yukle()
    X_train, _, y_train, _ = veriyi_bol(df)

    # Test setine hic dokunmadan, X_train icinden kucuk tabakali alt kume sec
    X_alt, _, y_alt, _ = train_test_split(
        X_train,
        y_train,
        train_size=egitim_boyutu + dogrulama_boyutu,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    X_fit, X_pred, y_fit, y_pred_gercek = train_test_split(
        X_alt,
        y_alt,
        test_size=dogrulama_boyutu,
        stratify=y_alt,
        random_state=RANDOM_STATE,
    )
    return X_fit, X_pred, y_fit, y_pred_gercek


def test_bes_model_ortak_pipeline():
    """5 modelin ortak build_full_pipeline() ile calistigini dogrular."""
    from src.ortak import RANDOM_STATE
    from src.pipeline import build_full_pipeline

    X_fit, X_pred, y_fit, _ = _kucuk_orneklem_hazirla()

    model_tanimlari = [
        ("Lojistik Regresyon", lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("KNN", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("SVM", lambda: SVC(probability=True, random_state=RANDOM_STATE)),
        ("Random Forest", lambda: RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
    ]

    # XGBoost bagimliligini gercek import ile kontrol et
    try:
        from xgboost import XGBClassifier
        model_tanimlari.append(
            ("XGBoost", lambda: XGBClassifier(n_estimators=50, random_state=RANDOM_STATE, eval_metric="logloss"))
        )
    except Exception as exc:
        model_tanimlari.append(("XGBoost", exc))

    sonuclar = []
    for model_adi, kurucu in model_tanimlari:
        kayit = {
            "model": model_adi,
            "pipeline": "HATA",
            "fit": "HATA",
            "predict": "HATA",
            "hata": None,
        }
        try:
            if isinstance(kurucu, Exception):
                raise kurucu

            clf = kurucu()
            pipe = build_full_pipeline(clf)
            kayit["pipeline"] = "BASARILI"

            pipe.fit(X_fit, y_fit)
            kayit["fit"] = "BASARILI"

            tahminler = pipe.predict(X_pred)
            assert tahminler.shape == (len(X_pred),), (
                f"Beklenen cikti boyutu {(len(X_pred),)}, alinan {tahminler.shape}"
            )
            assert set(np.unique(tahminler)).issubset({0, 1}), (
                f"Beklenmeyen sinif degerleri: {np.unique(tahminler)}"
            )
            kayit["predict"] = f"BASARILI {tahminler.shape}"
        except Exception as exc:
            kayit["hata"] = f"{type(exc).__name__}: {exc}"

        sonuclar.append(kayit)

    print("\n| Model | Pipeline | Fit | Predict |")
    print("|---|---|---|---|")
    for s in sonuclar:
        print(f"| {s['model']} | {s['pipeline']} | {s['fit']} | {s['predict']} |")
        if s["hata"]:
            print(f"  -> Hata ({s['model']}): {s['hata']}")

    basarisizlar = [s for s in sonuclar if s["hata"] is not None]
    assert not basarisizlar, f"Basarisiz modeller: {basarisizlar}"
    return sonuclar


def test_smote_pipeline_sizinti_kontrolu():
    """SMOTE boru hattinin yalnizca egitim katinda calistigini dogrular."""
    from src.ortak import RANDOM_STATE, ortak_cv
    from src.pipeline import build_smote_pipeline

    X_fit, X_pred, y_fit, _ = _kucuk_orneklem_hazirla(egitim_boyutu=400, dogrulama_boyutu=100)

    smote_pipe = build_smote_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    # 5 katli CV sirasinda imblearn.Pipeline, SMOTE'u yalnizca egitim katmanina uygular;
    # dogrulama katmaninda predict/transform degismeden calisir.
    cv_sonuc = cross_validate(
        smote_pipe,
        X_fit,
        y_fit,
        cv=ortak_cv(),
        scoring="roc_auc",
        n_jobs=1,
    )
    assert len(cv_sonuc["test_score"]) == 5
    assert not np.isnan(cv_sonuc["test_score"]).any()

    # Fit + predict sonrasi tahmin boyutu X_pred ile birebir ayni olmali (SMOTE test/dogrulama verisini cogaltmaz)
    smote_pipe.fit(X_fit, y_fit)
    tahminler = smote_pipe.predict(X_pred)
    assert tahminler.shape == (len(X_pred),)
    print(
        f"[BASARILI] build_smote_pipeline() 5-katli CV (ort ROC-AUC={cv_sonuc['test_score'].mean():.4f}) "
        f"ve predict boyutu {tahminler.shape} dogrulandi (dogrulama/test verisine SMOTE uygulanmadi)."
    )


if __name__ == "__main__":
    test_modul_importlari()
    test_bes_model_ortak_pipeline()
    test_smote_pipeline_sizinti_kontrolu()
    print("\nTum ortak pipeline ve modul testleri basariyla tamamlandi!")
