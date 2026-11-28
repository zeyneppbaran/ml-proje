"""Ortak Veri Yukleme, %70/%15/%15 Bolme, Pipeline ve 5 Model Uyumluluk Testi.

Bu test dosyasi asagidaki kontrollerin tamamini gercek calistirma ile dogrular:
1. Ortak Python modullerinin (ortak, preprocessing, pipeline) import edilmesi ve temel kurallar.
2. Ham verideki gercek tekrarlarin (duration sutunu henuz mevcutken) silinmesi.
3. daha_once_aranmis (previous > 0), pdays_kayitli (pdays != 999) ve pdays (999 -> -1) donusumleri.
4. duration sutununun model ozelliklerinden cikarilmasi.
5. %70 Egitim / %15 Dogrulama (Validation) / %15 Test tabakali (stratified) bolme oranlari ve sinif dagilimi.
6. Bes modelin (Lojistik Regresyon, KNN, SVM, Random Forest, XGBoost) ortak Pipeline ile fit ve predict testi.
7. 5 katli StratifiedKFold capraz dogrulama (yalnizca egitim verisinde).
8. SMOTE boru hattinin yalnizca egitim katlarinda calismasi ve dogrulama/test verisine uygulanmamasi.

Calistirma:
    python tests/test_ortak_pipeline.py
    pytest tests/test_ortak_pipeline.py -s
"""
import sys
import tempfile
from pathlib import Path

# Proje kok dizinini Python arama yoluna ekle
PROJE_KOKU = Path(__file__).resolve().parent.parent
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate


def test_modul_importlari_ve_on_isleme_kurallari():
    """Ortak modullerin import edildigini ve ColumnTransformer kurallarini dogrular."""
    import src
    import src.ortak
    import src.preprocessing
    import src.pipeline

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

    assert hasattr(src.preprocessing, "create_preprocessor")
    assert hasattr(src.preprocessing, "prepare_unknown_values")
    assert "daha_once_aranmis" in src.preprocessing.numeric_features
    assert "pdays_kayitli" in src.preprocessing.numeric_features

    prep = src.ortak.on_isleyici()
    transformers = dict((name, tr) for name, tr, _ in prep.transformers)
    assert transformers["num"].named_steps["imputer"].strategy == "median"
    assert transformers["num"].named_steps["scaler"].__class__.__name__ == "StandardScaler"
    assert transformers["mode"].named_steps["imputer"].strategy == "most_frequent"
    assert transformers["mode"].named_steps["encoder"].handle_unknown == "ignore"
    assert transformers["unknown"].named_steps["encoder"].handle_unknown == "ignore"
    assert transformers["other_cat"].named_steps["encoder"].handle_unknown == "ignore"
    assert src.preprocessing.unknown_features == ["default"]

    assert hasattr(src.pipeline, "build_full_pipeline")
    assert hasattr(src.pipeline, "build_smote_pipeline")

    cv = src.ortak.ortak_cv()
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42
    print("[BASARILI] Ortak moduller, ColumnTransformer kurallari ve 5-katli ortak_cv() dogrulandi.")


def test_veri_yukleme_tekrar_ve_ozellik_donusumleri():
    """Ham tekrar silme, daha_once_aranmis, pdays_kayitli, pdays ve unknown kurallarini dogrular."""
    import src.ortak
    import src.preprocessing

    ham_yol = src.ortak._bul_veri_yolu()
    ham_df = pd.read_csv(ham_yol, sep=";")
    ham_tekrar_sayisi = int(ham_df.duplicated().sum())
    ham_tekil_df = ham_df.drop_duplicates().reset_index(drop=True)

    df = src.ortak.veri_yukle()

    # 1. Tekrarlanan ham kayitlar duration henuz varken silinmis olmali ve sinif dengesi bozulmamali
    assert len(df) == len(ham_df) - ham_tekrar_sayisi, (
        f"Beklenen satir sayisi {len(ham_df) - ham_tekrar_sayisi}, alinan {len(df)}"
    )
    assert int(df.duplicated().sum()) == 0, "Tekrarlanan satir kalmis!"
    ham_y_orani = float((ham_df["y"] == "yes").mean())
    temiz_y_orani = float(df["y"].mean())
    assert abs(ham_y_orani - temiz_y_orani) < 0.001, "Tekrar silme hedef sinif dagilimini bozdu!"

    # 2. daha_once_aranmis gercekten previous > 0 kosuluna gore hesaplanmali
    beklenen_aranmis = (ham_tekil_df["previous"] > 0).astype(int)
    assert (df["daha_once_aranmis"].values == beklenen_aranmis.values).all(), (
        "daha_once_aranmis sutunu previous > 0 kosulu ile eslesmiyor!"
    )

    # 3. pdays_kayitli gercekten ham pdays != 999 kosuluna gore hesaplanmali
    beklenen_pdays_kayitli = (ham_tekil_df["pdays"] != 999).astype(int)
    assert (df["pdays_kayitli"].values == beklenen_pdays_kayitli.values).all(), (
        "pdays_kayitli sutunu pdays != 999 kosulu ile eslesmiyor!"
    )
    # Iki sutunun anlamlari farkli oldugundan degerleri de birebir ayni olmamali
    assert not (df["daha_once_aranmis"].values == df["pdays_kayitli"].values).all(), (
        "daha_once_aranmis ve pdays_kayitli birebir ayni cikti!"
    )

    # 4. pdays sutununda 999 kalmamis ve kayitsiz olanlar Pipeline icinde median ile doldurulmak uzere NaN yapilmis olmali
    assert int((df["pdays"] == 999).sum()) == 0, "pdays sutununda 999 degeri kalmis!"
    assert int(df["pdays"].isna().sum()) == int((ham_tekil_df["pdays"] == 999).sum()), (
        "pdays NaN sayisi ham pdays == 999 sayisi ile eslesmiyor!"
    )
    # Kayitli pdays degerleri degismemis olmali
    kayitli_mask = ham_tekil_df["pdays"] != 999
    assert (df.loc[kayitli_mask, "pdays"].values == ham_tekil_df.loc[kayitli_mask, "pdays"].values).all()

    # 5. default sutunundaki unknown korunuyor, mode_features sutunlarinda NaN yapiliyor
    assert int((df["default"] == "unknown").sum()) > 0, "default sutunundaki unknown korunmamis!"
    for col in src.preprocessing.mode_features:
        assert int((df[col] == "unknown").sum()) == 0, f"{col} sutununda unknown kalmis!"

    # 6. Hedef sutun yes=1, no=0 olarak kodlanmis olmali
    assert set(df["y"].unique()) == {0, 1}
    print(
        f"[BASARILI] Ham satir: {len(ham_df)}, silinen gercek tekrar: {ham_tekrar_sayisi}, "
        f"kalan: {len(df)} | daha_once_aranmis=1: {int(df['daha_once_aranmis'].sum())}, "
        f"pdays_kayitli=1: {int(df['pdays_kayitli'].sum())}."
    )


def test_veri_bolme_oranlari_sinif_dagilimi_ve_duration():
    """%70/%15/%15 tabakali bolme, sinif dagilimi ve duration cikarilmasini dogrular."""
    import src.ortak
    import src.preprocessing

    df = src.ortak.veri_yukle()
    X_train, X_val, X_test, y_train, y_val, y_test = src.ortak.veriyi_bol(df)

    # duration sutunu model ozelliklerinden cikarilmis olmali
    for ad, parca in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        assert "duration" not in parca.columns, f"{ad} icinde duration sutunu kalmis!"
        assert "y" not in parca.columns, f"{ad} icinde hedef y sutunu kalmis!"

    # Tum beklenen ozellikler mevcut olmali
    tum_beklenen = set(
        src.preprocessing.numeric_features
        + src.preprocessing.mode_features
        + src.preprocessing.unknown_features
        + src.preprocessing.other_cat_features
    )
    assert set(X_train.columns) == tum_beklenen, "X_train sutunlari preprocessing listeleriyle eslesmiyor!"

    # Toplam kayit ve cakismanin olmamasi
    toplam = len(df)
    assert len(X_train) + len(X_val) + len(X_test) == toplam
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    # Hesaplanan oranlar yaklasik %70 / %15 / %15 olmali
    oran_train = len(X_train) / toplam
    oran_val = len(X_val) / toplam
    oran_test = len(X_test) / toplam
    assert abs(oran_train - 0.70) < 0.005, f"Train orani beklenenden sapmis: {oran_train:.4f}"
    assert abs(oran_val - 0.15) < 0.005, f"Validation orani beklenenden sapmis: {oran_val:.4f}"
    assert abs(oran_test - 0.15) < 0.005, f"Test orani beklenenden sapmis: {oran_test:.4f}"

    # Uc veri kumesinde de pozitif sinif dagilimi korunmus olmali
    pozitif_genel = float(df["y"].mean())
    pozitif_train = float(y_train.mean())
    pozitif_val = float(y_val.mean())
    pozitif_test = float(y_test.mean())
    assert abs(pozitif_train - pozitif_genel) < 0.002
    assert abs(pozitif_val - pozitif_genel) < 0.002
    assert abs(pozitif_test - pozitif_genel) < 0.002

    print(
        f"[BASARILI] Bolme oranlari -> Train: {len(X_train)} (%{oran_train*100:.2f}, y=1 %{pozitif_train*100:.2f}), "
        f"Val: {len(X_val)} (%{oran_val*100:.2f}, y=1 %{pozitif_val*100:.2f}), "
        f"Test: {len(X_test)} (%{oran_test*100:.2f}, y=1 %{pozitif_test*100:.2f})."
    )


def test_on_isleyici_fit_transform_ve_notebook_uyumlulugu():
    """On isleyicinin NaN/Inf birakmadigini ve model notebook'larinin 6'li bolme kullandigini dogrular."""
    import json
    import src.ortak

    df = src.ortak.veri_yukle()
    X_train, X_val, X_test, _, _, _ = src.ortak.veriyi_bol(df)

    prep = src.ortak.on_isleyici()
    X_tr = prep.fit_transform(X_train)
    X_va = prep.transform(X_val)
    X_te = prep.transform(X_test)

    assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1]
    assert int(np.isnan(X_tr).sum()) == 0
    assert int(np.isnan(X_va).sum()) == 0
    assert int(np.isnan(X_te).sum()) == 0
    assert int(np.isinf(X_tr).sum()) == 0

    # Tum model/SMOTE/SHAP notebook'larinda eski 4'lu veriyi_bol kullanimi kalmamis olmali
    kontrol_edilecek = [
        "03_on_isleme.ipynb",
        "05_dengeleme_smote.ipynb",
        "06_model_lojistik_regresyon.ipynb",
        "07_model_knn.ipynb",
        "08_model_svm.ipynb",
        "09_model_random_forest.ipynb",
        "10_model_xgboost.ipynb",
        "11_aciklanabilirlik_shap.ipynb",
    ]
    for ad in kontrol_edilecek:
        with open(PROJE_KOKU / "notebooks" / ad, encoding="utf-8") as f:
            nb = json.load(f)
        tum_kod = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert "X_train, X_test, y_train, y_test = veriyi_bol" not in tum_kod, (
            f"{ad} icinde eski 4'lu veriyi_bol kullanimi kalmis!"
        )
        assert "X_train, X_val, X_test, y_train, y_val, y_test = veriyi_bol" in tum_kod, (
            f"{ad} icinde yeni 6'li veriyi_bol kullanimi bulunamadi!"
        )

    # 06_model_lojistik_regresyon.ipynb icinde X_test/y_test degiskenleri E1, E2, E4 hucrelerinde hic kullanilmamali (AST kontrolu)
    import ast
    with open(PROJE_KOKU / "notebooks" / "06_model_lojistik_regresyon.ipynb", encoding="utf-8") as f:
        nb06 = json.load(f)
    for idx, cell in enumerate(nb06["cells"]):
        if cell["cell_type"] == "code" and idx in (5, 8, 14):  # E1, E2, E4
            kod = "".join(cell["source"])
            degiskenler = {n.id for n in ast.walk(ast.parse(kod)) if isinstance(n, ast.Name)}
            assert "X_test" not in degiskenler and "y_test" not in degiskenler, (
                f"06_model_lojistik_regresyon.ipynb hucre {idx} icinde X_test/y_test kullanimi tespit edildi!"
            )

    print(
        f"[BASARILI] On isleyici ({X_tr.shape[1]} ozellik, 0 NaN/Inf) ve 8 notebook'un 6'li veriyi_bol uyumlulugu dogrulandi."
    )


def _kucuk_orneklem_hazirla(egitim_boyutu=500, dogrulama_boyutu=100):
    """Yalnizca X_train ve y_train icinden kucuk tabakali egitim/tahmin orneklemi olusturur."""
    from src.ortak import veri_yukle, veriyi_bol, RANDOM_STATE

    df = veri_yukle()
    X_train, _, _, y_train, _, _ = veriyi_bol(df)

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
    """5 modelin (LR, KNN, SVM, RF, XGBoost) ortak build_full_pipeline() ile calistigini dogrular."""
    from src.ortak import RANDOM_STATE
    from src.pipeline import build_full_pipeline

    X_fit, X_pred, y_fit, _ = _kucuk_orneklem_hazirla()

    model_tanimlari = [
        ("Lojistik Regresyon", lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("KNN", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("SVM", lambda: SVC(random_state=RANDOM_STATE)),
        ("Random Forest", lambda: RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
    ]

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


def test_capraz_dogrulama_ve_degerlendirme_yardimcilari():
    """5 katli capraz dogrulama, degerlendir() ve sonuclari_oku() fonksiyonlarini dogrular."""
    import src.ortak
    from src.pipeline import build_full_pipeline

    X_fit, X_pred, y_fit, y_pred_gercek = _kucuk_orneklem_hazirla(egitim_boyutu=300, dogrulama_boyutu=100)
    ornek_pipe = build_full_pipeline(LogisticRegression(max_iter=1000, random_state=src.ortak.RANDOM_STATE))

    cv_ozet = src.ortak.capraz_dogrula(ornek_pipe, X_fit, y_fit, n_jobs=1)
    assert set(cv_ozet.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for metrik, deger in cv_ozet.items():
        assert len(deger["kat_skorlari"]) == 5, f"{metrik} icin 5 kat skoru donmedi!"
        assert not np.isnan(deger["ortalama"])

    ornek_pipe.fit(X_fit, y_fit)
    with tempfile.TemporaryDirectory() as gecici_klasor:
        sonuc = src.ortak.degerlendir(ornek_pipe, X_pred, y_pred_gercek, "test_kontrol", klasor=gecici_klasor)
        tablo = src.ortak.sonuclari_oku(klasor=gecici_klasor)
        assert len(tablo) == 1 and tablo.iloc[0]["model"] == "test_kontrol"
        assert "roc_auc" in sonuc

    print(
        f"[BASARILI] 5-katli capraz_dogrula() (ort ROC-AUC={cv_ozet['roc_auc']['ortalama']:.4f}) "
        "ve degerlendir()/sonuclari_oku() dogrulandi."
    )


def test_smote_pipeline_sizinti_kontrolu():
    """SMOTE boru hattinin yalnizca egitim katlarinda calistigini dogrular."""
    from src.ortak import RANDOM_STATE, ortak_cv
    from src.pipeline import build_smote_pipeline

    X_fit, X_pred, y_fit, _ = _kucuk_orneklem_hazirla(egitim_boyutu=400, dogrulama_boyutu=100)

    smote_pipe = build_smote_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )

    # 5 katli CV sirasinda imblearn.Pipeline SMOTE'u yalnizca ilgili egitim katinda calistirir
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

    # Fit sirasinda SMOTE yalnizca egitim verisini cogaltir; predict asamasinda X_pred boyutu degismez
    smote_pipe.fit(X_fit, y_fit)
    X_egitim_islenmis = smote_pipe.named_steps["preprocessor"].transform(X_fit)
    _, y_smote_sonrasi = smote_pipe.named_steps["smote"].fit_resample(X_egitim_islenmis, y_fit)
    assert (y_smote_sonrasi == 0).sum() == (y_smote_sonrasi == 1).sum(), "SMOTE egitim siniflarini dengelemedi!"

    tahminler = smote_pipe.predict(X_pred)
    assert tahminler.shape == (len(X_pred),), "SMOTE tahmin/dogrulama verisi boyutunu bozdu!"
    print(
        f"[BASARILI] build_smote_pipeline() 5-katli CV (ort ROC-AUC={cv_sonuc['test_score'].mean():.4f}) "
        f"ve predict boyutu {tahminler.shape} dogrulandi (dogrulama/test verisine SMOTE uygulanmadi)."
    )


if __name__ == "__main__":
    test_modul_importlari_ve_on_isleme_kurallari()
    test_veri_yukleme_tekrar_ve_ozellik_donusumleri()
    test_veri_bolme_oranlari_sinif_dagilimi_ve_duration()
    test_on_isleyici_fit_transform_ve_notebook_uyumlulugu()
    test_bes_model_ortak_pipeline()
    test_capraz_dogrulama_ve_degerlendirme_yardimcilari()
    test_smote_pipeline_sizinti_kontrolu()
    print("\nTum ortak veri isleme, %70/%15/%15 bolme, pipeline ve SMOTE testleri basariyla tamamlandi!")
