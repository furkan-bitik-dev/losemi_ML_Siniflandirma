# Veri setlerini okuyan fonksiyonlar.
# Üç EDA betiği de verileri buradan okuyor, böylece okuma kodunu tekrar yazmıyorum.
import gzip
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

# PyCharm conda ortamını "activate" etmeden çalıştırıyor. cuML'in GPU kütüphanelerini
# bulabilmesi için ortamın yolunu burada bir kere veriyorum (EDA dosyaları önce bu dosyayı import ediyor).
os.environ.setdefault("CONDA_PREFIX", sys.prefix)
os.environ.setdefault("CUDA_PATH", sys.prefix)

# Klasörler (bu dosya src/ içinde olduğu için bir üst klasöre çıkıyorum)
PROJE = Path(__file__).resolve().parent.parent
VERI = PROJE / "data" / "raw"          # ham veriler
SEKIL = PROJE / "sonuclar" / "sekiller"  # şekillerin kaydedileceği yer
TABLO = PROJE / "sonuclar" / "tablolar"  # gen listelerinin kaydedileceği yer
SEKIL.mkdir(parents=True, exist_ok=True)
TABLO.mkdir(parents=True, exist_ok=True)


def golub_dosya_oku(dosya):
    """Golub CSV dosyasını okur. X: ifade değerleri, C: P/A/M çağrıları (satır = hasta)."""
    df = pd.read_csv(dosya)
    # İlk iki sütun gen açıklaması ve prob numarası.
    # Sonra her hasta için iki sütun var: ifade değeri ve "call" (P/A/M).
    hasta_sutunlari = [c for c in df.columns[2:] if not c.startswith("call")]
    call_sutunlari = [c for c in df.columns[2:] if c.startswith("call")]

    X = df[hasta_sutunlari].T.astype(float)
    X.index = X.index.astype(int)                    # satır adı = hasta numarası
    X.columns = df["Gene Accession Number"].values   # sütun adı = prob numarası

    C = df[call_sutunlari].T
    C.index = X.index
    C.columns = X.columns
    return X, C


def golub_oku():
    """Eğitim ve test dosyalarını birleştirir, etiketleri hasta numarasıyla eşler."""
    X_egitim, C_egitim = golub_dosya_oku(VERI / "golub" / "data_set_ALL_AML_train.csv")
    X_test, C_test = golub_dosya_oku(VERI / "golub" / "data_set_ALL_AML_independent.csv")
    etiket = pd.read_csv(VERI / "golub" / "actual.csv").set_index("patient")["cancer"]

    X = pd.concat([X_egitim, X_test])
    C = pd.concat([C_egitim, C_test])
    # DİKKAT: dosyalardaki sütunlar hasta numarası sırasında değil.
    # Bu yüzden etiketi sıraya göre değil hasta numarasına göre alıyorum.
    y = etiket.loc[X.index]
    set_bilgisi = pd.Series(["Eğitim"] * len(X_egitim) + ["Test"] * len(X_test), index=X.index)
    return X, C, y, set_bilgisi, etiket


def gse_oku():
    """GSE7186 series matrix dosyasını okur. X: prob x örnek, grup: her örneğin grubu."""
    dosya = VERI / "gse7186" / "GSE7186_series_matrix.txt.gz"
    # Dosyanın başında "!Sample_..." ile başlayan örnek bilgileri,
    # sonra table_begin ile table_end arasında ifade tablosu var.
    bilgi = {}
    tablo = []
    tablo_icinde = False
    with gzip.open(dosya, "rt", encoding="utf-8", errors="replace") as f:
        for satir in f:
            satir = satir.rstrip("\n")
            if satir.startswith("!series_matrix_table_begin"):
                tablo_icinde = True
            elif satir.startswith("!series_matrix_table_end"):
                break
            elif tablo_icinde:
                tablo.append(satir.split("\t"))
            elif satir.startswith("!Sample_title") or satir.startswith("!Sample_geo_accession"):
                parcalar = satir.split("\t")
                bilgi[parcalar[0][1:]] = [p.strip('"') for p in parcalar[1:]]

    ornekler = [h.strip('"') for h in tablo[0][1:]]
    problar = [r[0].strip('"') for r in tablo[1:]]
    X = pd.DataFrame([r[1:] for r in tablo[1:]], index=problar, columns=ornekler)
    X = X.apply(pd.to_numeric, errors="coerce")   # boş hücreler NaN (eksik değer) oluyor

    # Matristeki sütunlar ile örnek bilgileri aynı sırada mı? (GSM numarasıyla kontrol)
    assert bilgi["Sample_geo_accession"] == ornekler, "GSM numaraları eşleşmiyor!"

    # Grup bilgisi örnek başlığında yazıyor: "B-lineage ALL 12" gibi. Sondaki numarayı atıyorum.
    baslik = pd.Series(bilgi["Sample_title"], index=ornekler).str.replace(r"\s*\d+$", "", regex=True)
    grup = baslik.map({"B-lineage ALL": "B-ALL", "T-cell ALL": "T-ALL",
                       "AML": "AML", "Normal Bone Marrow": "Normal KI"})
    return X, grup


def anotasyon_oku(dosya):
    """GEO .annot.gz dosyasından prob numarası -> gen sembolü eşlemesini okur."""
    with gzip.open(dosya, "rt", encoding="utf-8", errors="replace") as f:
        satirlar = f.readlines()
    bas = next(i for i, s in enumerate(satirlar) if s.startswith("!platform_table_begin"))
    son = next(i for i, s in enumerate(satirlar) if s.startswith("!platform_table_end"))
    df = pd.read_csv(StringIO("".join(satirlar[bas + 1:son])), sep="\t", dtype=str)
    return df.set_index("ID")["Gene symbol"]
