# Golub ve GSE7186 veri setlerinin karşılaştırılması
# Soru: Farklı teknolojiyle ölçülmüş iki veri seti birlikte kullanılabilir mi? ALL-AML farkı ikisinde de aynı genlerde mi?
# Çalıştırma: PyCharm'da bu dosyayı Run ile çalıştır (önce veri_indir.py).
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from veri_okuma import VERI, SEKIL, golub_oku, gse_oku, anotasyon_oku

# GPU üzerinde çalışan RAPIDS cuML (kullanımı scikit-learn ile aynı)
import cuml
from cuml.decomposition import PCA
from cuml.preprocessing import StandardScaler
from cuml.metrics.cluster import silhouette_score

cuml.set_global_output_type("numpy")   # cuML sonuçları numpy dizisi olarak dönsün
warnings.filterwarnings("ignore")      # sabit değerli genlerde scipy uyarı veriyor, sonucu etkilemiyor
renk = {"ALL": "tab:blue", "AML": "tab:orange", "Normal": "black"}


def sembole_cevir(X, sembol):
    # İki çipin prob numaraları farklı, ortak dil gen sembolü.
    # X: örnek x prob. Birden fazla gene karşılık gelen probları ("///" olanlar) atıyorum,
    # aynı geni ölçen birden fazla prob varsa ortalamasını alıyorum.
    s = sembol.reindex(X.columns.astype(str))
    s = s[s.notna() & ~s.str.contains("///", na=False)]
    return X[s.index].T.groupby(s.values).mean().T


# ================================================================ 1) İKİ VERİYİ OKU, ORTAK GENLERİ BUL
print("\n=== 1) Ortak genler ===")
X_golub, _, y_golub, _, _ = golub_oku()
G = np.log2(X_golub.clip(100, 16000))     # Golub: eda_golub.py'deki gibi kırpma + log2 (tüm problar)
X_gse, grup = gse_oku()
E = X_gse.T                               # GSE7186: satır = örnek olsun

G = sembole_cevir(G, anotasyon_oku(VERI / "golub" / "GPL80.annot.gz"))
E = sembole_cevir(E, anotasyon_oku(VERI / "gse7186" / "GPL4861.annot.gz"))
ortak = sorted(set(G.columns) & set(E.columns))
print("Golub gen sayısı:", G.shape[1], " GSE7186 gen sayısı:", E.shape[1], " ortak gen:", len(ortak))
# GSE7186'da eksiği %20'den fazla olan genleri çıkarıyorum
ortak = [g for g in ortak if E[g].isna().mean() <= 0.2]
print("GSE7186'da eksiği %20 ve altında olan ortak gen:", len(ortak))

G = G[ortak]
E = E[ortak].fillna(E[ortak].median())    # kalan eksikleri gen medyanıyla doldur (sadece bu analiz için)

# ================================================================ 2) İKİ VERİ SETİ BİRLİKTE KULLANILABİLİR Mİ?
print("\n=== 2) Dağılım farkı ===")
print(f"Golub (log2 şiddet): medyan {np.median(G.values):.2f}   GSE7186 (log2 oran): medyan {np.median(E.values):.2f}")

veri_seti = np.array(["Golub"] * len(G) + ["GSE7186"] * len(E))
sinif = np.concatenate([y_golub.values,
                        grup.map({"B-ALL": "ALL", "T-ALL": "ALL", "AML": "AML", "Normal KI": "Normal"}).values])

# (a) İki veriyi alt alta koyup BİRLİKTE ölçeklersem ne olur?
birlikte = pd.concat([G, E], ignore_index=True)
pca_a = PCA(n_components=2)
Q_a = pca_a.fit_transform(StandardScaler().fit_transform(birlikte.values.astype(np.float32)))

# (b) Her veri setini KENDİ İÇİNDE z-skora çevirip sonra birleştirirsem ne olur?
z_golub = (G - G.mean()) / G.std()
z_gse = (E - E.mean()) / E.std()
ayri = pd.concat([z_golub, z_gse], ignore_index=True).fillna(0)
pca_b = PCA(n_components=2)
Q_b = pca_b.fit_transform(ayri.values.astype(np.float32))

# Siluet değeri (-1 ile 1 arası): 1'e yakınsa gruplar birbirinden tamamen ayrı demek.
# Veri setine göre siluet yüksekse örnekler hastalığa göre değil geldikleri veri setine göre ayrılıyor demektir.
losemi = sinif != "Normal"
sil_a = silhouette_score(Q_a, pd.factorize(veri_seti)[0])
sil_b = silhouette_score(Q_b, pd.factorize(veri_seti)[0])
sil_sinif = silhouette_score(Q_b[losemi], pd.factorize(sinif[losemi])[0])
print(f"Birlikte ölçekleme  : PC1 {pca_a.explained_variance_ratio_[0]:.1%}, veri seti siluet {sil_a:.2f}")
print(f"Veri seti içi z-skor: PC1 {pca_b.explained_variance_ratio_[0]:.1%}, veri seti siluet {sil_b:.2f}, "
      f"sınıf siluet {sil_sinif:.2f}")
# Sonuç: birlikte ölçeklersem PC1 sadece veri setini ayırıyor. Her veriyi kendi içinde ölçeklemem gerekiyor.

# ================================================================ 3) ALL-AML FARKI İKİ VERİDE AYNI GENLERDE Mİ?
print("\n=== 3) Sınıf farkının uyumu ===")
y_gse = grup.map({"B-ALL": "ALL", "T-ALL": "ALL", "AML": "AML"}).dropna()   # sağlıklı örnekler hariç
E_losemi = E.loc[y_gse.index]
# her gen için AML - ALL farkının t istatistiği, iki veri setinde ayrı ayrı
t_golub, p_golub = stats.ttest_ind(G[y_golub.values == "AML"], G[y_golub.values == "ALL"], equal_var=False)
t_gse, p_gse = stats.ttest_ind(E_losemi[y_gse == "AML"], E_losemi[y_gse == "ALL"], equal_var=False)
t_golub, t_gse = pd.Series(t_golub, index=ortak), pd.Series(t_gse, index=ortak)
p_golub, p_gse = pd.Series(p_golub, index=ortak), pd.Series(p_gse, index=ortak)
gecerli = t_golub.notna() & t_gse.notna()        # sabit genlerde t hesaplanamıyor, onları çıkar
t_golub, t_gse, p_golub, p_gse = t_golub[gecerli], t_gse[gecerli], p_golub[gecerli], p_gse[gecerli]

rho = stats.spearmanr(t_golub, t_gse).statistic
anlamli_golub = set(t_golub.index[multipletests(p_golub, method="fdr_bh")[0]])
anlamli_gse = set(t_gse.index[multipletests(p_gse, method="fdr_bh")[0]])
ikisinde = sorted(anlamli_golub & anlamli_gse)   # iki veride de anlamlı (FDR < 0,05) genler
ayni_yon = sum(np.sign(t_golub[g]) == np.sign(t_gse[g]) for g in ikisinde)
print("t istatistiklerinin ilişkisi (Spearman rho):", round(rho, 2))
print("Anlamlı gen -> Golub:", len(anlamli_golub), " GSE7186:", len(anlamli_gse), " ikisinde de:", len(ikisinde))
print(f"İkisinde de anlamlı genlerden farkı aynı yönde olan: {ayni_yon} ({ayni_yon / len(ikisinde):.1%})")
ilk50_golub = set(t_golub.abs().sort_values(ascending=False).index[:50])
ilk50_gse = set(t_gse.abs().sort_values(ascending=False).index[:50])
print("İki verinin ilk 50 listesindeki ortak genler:", sorted(ilk50_golub & ilk50_gse))

# Şekil: (a) birlikte ölçekleme, (b) veri seti içi z-skor, (c) iki verideki t istatistikleri
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
for a, Q, pca in ((ax[0], Q_a, pca_a), (ax[1], Q_b, pca_b)):
    for s in ("ALL", "AML", "Normal"):
        m = sinif == s
        a.scatter(Q[m & (veri_seti == "Golub"), 0], Q[m & (veri_seti == "Golub"), 1],
                  s=15, marker="o", color=renk[s], label=s + " (Golub)")
        a.scatter(Q[m & (veri_seti == "GSE7186"), 0], Q[m & (veri_seti == "GSE7186"), 1],
                  s=15, marker="^", color=renk[s], label=s + " (GSE7186)")
    a.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax[0].set_title("(a) Birlikte ölçekleme")
ax[1].set_title("(b) Veri seti içinde z-skor")
ax[1].legend(fontsize=6)
ax[2].scatter(t_golub, t_gse, s=3, color="lightgray", label="diğer genler")
ax[2].scatter(t_golub[ikisinde], t_gse[ikisinde], s=4, color="purple", label="iki veride anlamlı")
ax[2].axhline(0, color="black", lw=0.5)
ax[2].axvline(0, color="black", lw=0.5)
ax[2].set(title=f"(c) AML-ALL farkı (Spearman rho = {rho:.2f})", xlabel="Golub t", ylabel="GSE7186 t")
ax[2].legend(fontsize=7)
fig.savefig(SEKIL / "karsilastirma_1.pdf")
plt.close(fig)

print("\nBitti. Şekil 'sonuclar/sekiller' klasöründe.")
