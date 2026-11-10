# ALL/AML Sınıflandırması – Keşifsel Veri Analizi

Dönem projemin kod deposu. Ara rapor aşamasında iki veri setinin keşifsel veri analizini (EDA) yaptım.
Modelleme kodunu final aşamasında ekleyeceğim.

## Veri setleri

| Veri | Kaynak | Boyut |
|---|---|---|
| Golub (ana veri) | Golub ve ark., 1999; ders modülündeki Kaggle sürümü | 72 hasta × 7129 prob (eğitim 38, test 34) |
| GSE7186 (benzer veri) | Andersson ve ark., 2007; NCBI GEO | 127 örnek × 21370 prob |

Ham veriler depoda yok. `src/veri_indir.py` dosyaları `data/raw/` klasörüne indiriyor.

## Nasıl çalıştırılır

Hesaplamaları ekran kartında (NVIDIA RTX 2050) RAPIDS cuML ile yaptım. RAPIDS Windows'ta doğrudan
çalışmadığı için WSL2 (Ubuntu) kullandım.

1. Ortamı kur (Ubuntu terminalinde, bir kere): `bash wsl_rapids_kurulum.sh`
2. PyCharm'da yorumlayıcı olarak WSL'deki `~/miniforge3/envs/rapids/bin/python` seç.
3. Dosyaları sırayla Run ile çalıştır:

| Sıra | Dosya | Ne yapıyor |
|---|---|---|
| 1 | `src/veri_indir.py` | Verileri indirir (varsa tekrar indirmez) |
| 2 | `src/eda_golub.py` | Golub: etiket kontrolü, veri kalitesi, ön işleme, PCA/t-SNE, ayırt edici genler, eğitim–test farkı |
| 3 | `src/eda_gse7186.py` | GSE7186: eksik değerler, PCA/t-SNE, ayırt edici genler |
| 4 | `src/eda_karsilastirma.py` | İki verinin ortak genleri, ölçek farkı, ALL–AML farkının uyumu |

`src/veri_okuma.py` verileri okuyan fonksiyonları içeriyor. Üç EDA dosyası birbirinden bağımsız;
istenen sırayla çalıştırılabilir.

## Çıktılar

- `sonuclar/sekiller/` : şekiller (PDF)
- `sonuclar/tablolar/` : ayırt edici gen listeleri (CSV)
- Sayısal sonuçlar çalıştırma sırasında konsola yazılıyor.

## Dikkat ettiğim noktalar

- Golub dosyalarında hasta sütunları sıralı değil. Etiketleri `actual.csv` ile hasta numarası üzerinden eşledim
  (sıraya göre eşleseydim 34 test hastasının 14'ü yanlış etiketlenirdi).
- Ön işleme: 100–16000 arasına kırpma, az değişen genleri eleme, log2 (Dudoit ve ark., 2002) → 3571 gen.
- Rastgelelik içeren adımlarda (t-SNE) tohum 42.
- EDA'daki ölçekleme ve gen testleri sadece veriyi tanımak için tüm örneklerde yapıldı. Modellemede
  gen seçimi, ölçekleme ve eksik değer doldurma sadece eğitim katlarında yapılacak (veri sızıntısı olmaması için).
