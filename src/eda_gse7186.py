# GSE7186 (Andersson ve ark., 2007) çocuk lösemisi verisi - Keşifsel Veri Analizi (EDA)
# Çalıştırma: PyCharm'da bu dosyayı Run ile çalıştır (önce veri_indir.py).
#
# NOT: Eksik değerleri burada sadece görselleştirme için gen medyanıyla doldurdum.
#      Modellemede doldurma çapraz doğrulama katlarının içinde, eğitim verisinin medyanıyla yapılacak.
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.multitest import multipletests

from veri_okuma import VERI, SEKIL, TABLO, gse_oku, anotasyon_oku

# GPU üzerinde çalışan RAPIDS cuML (kullanımı scikit-learn ile aynı)
import cuml
from cuml import PCA, TSNE
from cuml.preprocessing import StandardScaler

cuml.set_global_output_type("numpy")   # cuML sonuçları numpy dizisi olarak dönsün
warnings.filterwarnings("ignore")      # sabit değerli genlerde scipy uyarı veriyor, sonucu etkilemiyor
gruplar = ["B-ALL", "T-ALL", "AML", "Normal KI"]
renk = {"B-ALL": "tab:blue", "T-ALL": "tab:green", "AML": "tab:orange", "Normal KI": "black"}

# ================================================================ 1) VERİYİ OKU
print("\n=== 1) Veri yükleme ===")
X, grup = gse_oku()     # X: prob x örnek (log2 oran), grup: her örneğin grubu
print("Prob x örnek:", X.shape)
print("Grup sayıları:", grup.value_counts().to_dict())
print("Etiketi bulunamayan örnek:", grup.isna().sum())

# ================================================================ 2) VERİ KALİTESİ: eksik değerler
print("\n=== 2) Veri kalitesi ===")
eksik_ornek = X.isna().mean(axis=0)    # her örnekte eksik değer oranı
eksik_prob = X.isna().mean(axis=1)     # her probda eksik değer oranı
print(f"Toplam eksik oranı: {np.isnan(X.values).mean():.1%}   INF: {np.isinf(X.values).sum()}")
print(f"Örnek başına eksik: en az {eksik_ornek.min():.1%}, en çok {eksik_ornek.max():.1%}")

# Eksiklik gruplara göre değişiyor mu? Değişiyorsa eksiklik sınıf hakkında bilgi taşır, bu sorun olurdu.
p_kw = stats.kruskal(*[eksik_ornek[grup == g] for g in gruplar]).pvalue
print("Gruplar arasında eksik oranı farkı (Kruskal-Wallis) p =", round(p_kw, 3))
print("Hiç eksiği olmayan prob:", (eksik_prob == 0).sum(), " | %20'den fazla eksiği olan:", (eksik_prob > 0.2).sum())
print("Değer aralığı:", round(np.nanmin(X.values), 2), "ile", round(np.nanmax(X.values), 2))

# Aykırı örnek var mı? Her örneğin medyanına ve yayılımına (IQR) bakıyorum.
# Diğerlerinden 3 MAD'den fazla sapan örnekleri aday olarak işaretliyorum (silmiyorum).
medyan = X.median()
iqr = X.quantile(0.75) - X.quantile(0.25)
print("Örnek medyanları:", round(medyan.min(), 3), "ile", round(medyan.max(), 3), "arasında")
z_medyan = (medyan - medyan.median()) / stats.median_abs_deviation(medyan, scale="normal")
z_iqr = (iqr - iqr.median()) / stats.median_abs_deviation(iqr, scale="normal")
aykiri = list(X.columns[(z_medyan.abs() > 3) | (z_iqr.abs() > 3)])
print("Aykırı aday örnekler:", aykiri, "->", grup[aykiri].tolist())

# ================================================================ 3) BOYUT İNDİRGEME: PCA ve t-SNE (GPU)
print("\n=== 3) PCA ve t-SNE ===")
# PCA eksik değerle çalışmıyor. Eksiği %20'den az olan probları alıp kalan eksikleri gen medyanıyla dolduruyorum.
X_dolu = X[eksik_prob <= 0.2].T          # artık satır = örnek
X_dolu = X_dolu.fillna(X_dolu.median())
en_degisken = X_dolu.var().sort_values(ascending=False).index[:2000]   # en çok değişen 2000 prob
Z = StandardScaler().fit_transform(X_dolu[en_degisken].values.astype(np.float32))

pca = PCA(n_components=30).fit(Z)
oran = pca.explained_variance_ratio_
P = pca.transform(Z)
print("PC1, PC2, PC3 açıklanan varyans:", oran[:3].round(3))
ts = TSNE(n_components=2, perplexity=20, method="exact", random_state=42).fit_transform(P)

# Şekil: (a) örnek başına eksik, (b) prob başına eksik, (c) PCA, (d) t-SNE
fig, ax = plt.subplots(2, 2, figsize=(7, 6), layout="constrained")
ax[0, 0].boxplot([eksik_ornek[grup == g] * 100 for g in gruplar])
ax[0, 0].set_xticklabels(["B-ALL", "T-ALL", "AML", "Normal"])
ax[0, 0].set(title="(a) Örnek başına eksik", ylabel="Eksik değer (%)")
ax[0, 1].hist(eksik_prob * 100, bins=25, color="gray")
ax[0, 1].axvline(20, color="black", ls="--", label="%20 eşiği")
ax[0, 1].set(title="(b) Prob başına eksik", xlabel="Eksik (%)", ylabel="Prob sayısı")
ax[0, 1].legend()
for a, M in ((ax[1, 0], P), (ax[1, 1], ts)):
    for g in gruplar:
        m = (grup == g).values
        a.scatter(M[m, 0], M[m, 1], s=15, color=renk[g], label=g)
ax[1, 0].set(title="(c) PCA", xlabel=f"PC1 ({oran[0]:.1%})", ylabel=f"PC2 ({oran[1]:.1%})")
ax[1, 1].set(title="(d) t-SNE", xlabel="t-SNE 1", ylabel="t-SNE 2")
ax[1, 0].legend(fontsize=7)
fig.savefig(SEKIL / "gse_1_kalite_pca.pdf")
plt.close(fig)

# ================================================================ 4) AYIRT EDİCİ GENLER (ALL ve AML)
print("\n=== 4) Ayırt edici genler ===")
# Golub ile karşılaştırabilmek için B-ALL ve T-ALL'yi "ALL" olarak birleştiriyorum. Sağlıklı örnekler hariç.
sinif = grup.map({"B-ALL": "ALL", "T-ALL": "ALL", "AML": "AML"}).dropna()
X_test = X[eksik_prob <= 0.2][sinif.index].T
all_, aml = X_test[sinif == "ALL"], X_test[sinif == "AML"]
t, p = stats.ttest_ind(aml, all_, equal_var=False, nan_policy="omit")   # Welch t-testi, eksikleri atlayarak
t, p = np.asarray(t), np.asarray(p)
q = multipletests(p, method="fdr_bh")[1]                               # çoklu test düzeltmesi (FDR)

sembol = anotasyon_oku(VERI / "gse7186" / "GPL4861.annot.gz")
genler = pd.DataFrame({"prob": X_test.columns, "sembol": sembol.reindex(X_test.columns.astype(str)).values,
                       "t": t, "p": p, "q": q, "log2_fark": (aml.mean() - all_.mean()).values}).sort_values("p")
genler.to_csv(TABLO / "gse7186_ayirt_edici_genler.csv", index=False)
print("Test edilen prob:", len(genler), " q < 0,05:", (q < 0.05).sum())
print("En ayırt edici 10 gen:", genler.sembol.head(10).fillna("-").tolist())

# En iyi 30 probla iki küme: kümeler ALL ve AML ile örtüşüyor mu?
en_iyi = genler.prob[:30]
H = X_test[en_iyi].fillna(X_test[en_iyi].median())
Hz = (H - H.mean()) / H.std()
kume = fcluster(linkage(Hz.values, "average", metric="correlation"), 2, "maxclust")
print(pd.crosstab(kume, grup[sinif.index].values))
print(f"Küme saflığı (ALL/AML): {pd.crosstab(kume, sinif.values).max(axis=1).sum() / len(sinif):.1%}")
# T-ALL örnekleri AML ile aynı kümeye düşüyor: listenin başındaki genler B hücre genleri olduğu için.

print("\nBitti. Şekiller ve tablolar 'sonuclar' klasöründe.")
