# Ham verileri data/raw klasörüne indirir. Dosya zaten varsa tekrar indirmez.
# Çalıştırma: PyCharm'da bu dosyayı Run ile çalıştır.
import urllib.request

from veri_okuma import VERI

GITHUB = "https://raw.githubusercontent.com/dharsandip/Classification_of_Cancer_by_Gene_Expression_Dataset/main/"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/"

# kaydedilecek yer : indirme adresi
dosyalar = {
    VERI / "golub" / "data_set_ALL_AML_train.csv": GITHUB + "data_set_ALL_AML_train.csv",
    VERI / "golub" / "data_set_ALL_AML_independent.csv": GITHUB + "data_set_ALL_AML_independent.csv",
    VERI / "golub" / "actual.csv": GITHUB + "actual.csv",
    VERI / "golub" / "GPL80.annot.gz": GEO + "platforms/GPLnnn/GPL80/annot/GPL80.annot.gz",
    VERI / "gse7186" / "GSE7186_series_matrix.txt.gz": GEO + "series/GSE7nnn/GSE7186/matrix/GSE7186_series_matrix.txt.gz",
    VERI / "gse7186" / "GPL4861.annot.gz": GEO + "platforms/GPL4nnn/GPL4861/annot/GPL4861.annot.gz",
}

for yol, adres in dosyalar.items():
    yol.parent.mkdir(parents=True, exist_ok=True)
    if yol.exists():
        print("zaten var:", yol.name)
    else:
        print("indiriliyor:", adres)
        urllib.request.urlretrieve(adres, yol)
print("Bitti.")
