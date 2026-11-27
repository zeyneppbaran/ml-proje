# Makine Öğrenmesi Dersi Proje Ödevi

**Proje adı:** (veri seti seçilince yazılacak)

## Ekip

| Kişi | Ad Soyad | Öğrenci No | Ortak paket | Model |
|---|---|---|---|---|
| 1 | | | Veri seti tanıtımı + özellik mühendisliği/seçimi | SVM |
| 2 | Aybüke Gökçe Yavaş | B221210371 | Keşif analizi ve grafikler | XGBoost |
| 3 | | | Ön işleme ve ortak pipeline | Lojistik Regresyon |
| 4 | Zeynep Baran | B231210029 | Açıklanabilir yapay zeka (SHAP) | Random Forest |
| 5 | | | Dengeleme (SMOTE) + karşılaştırma ve sonuç | KNN |

## Klasörler

- `notebooks/` : numara sırasıyla çalıştırılan Jupyter notebook'ları. Her notebook'un tek bir sahibi vardır.
- `src/ortak.py` : tüm notebook'ların kullandığı ortak fonksiyonlar (veri yükleme, bölme, ön işleme, değerlendirme). Sadece Kişi 3 değiştirir.
- `results/` : her modelin metrikleri (json).
- `figures/` : rapora girecek grafikler (png).
- `rapor/` : Word raporu ve sunum.

## Ortak kurallar

- Her yerde `random_state=42`.
- Test oranı: %20, `stratify=y`.
- Test seti sadece en sonda, bir kez kullanılır. Hiperparametreler cross validation ile seçilir.

## Colab'de çalışma

1. colab.research.google.com > Dosya > Not defterini aç > GitHub sekmesi > bu repoyu seç > kendi notebook'unu aç.
2. En üstteki "Kurulum" hücresini çalıştır (repoyu Colab'e indirir).
3. Çalış. Bitince: Dosya > Kopyasını GitHub'a kaydet.
