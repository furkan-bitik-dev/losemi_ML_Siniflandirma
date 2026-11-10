# Golub ALL/AML verisi - Keşifsel Veri Analizi (EDA)
# Çalıştırma: PyCharm'da bu dosyayı Run ile çalıştır (önce veri_indir.py).
#
# NOT: Buradaki ölçekleme ve gen testleri tüm 72 hastada yapıldı, çünkü amaç sadece veriyi tanımak.
#      Modellemede bu adımların hepsi çapraz doğrulama katlarının içinde, sadece eğitim verisiyle yapılacak.
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.multitest import multipletests

from veri_okuma import VERI, SEKIL, TABLO, golub_oku, anotasyon_oku

# GPU üzerinde çalışan RAPIDS cuML (kullanımı scikit-learn ile aynı)
import cuml
from cuml import PCA, TSNE
from cuml.preprocessing import StandardScaler

cuml.set_global_output_type("numpy")   # cuML sonuçları numpy dizisi olarak dönsün
warnings.filterwarnings("ignore")      # sabit değerli genlerde scipy uyarı veriyor, sonucu etkilemiyor
renk = {"ALL": "tab:blue", "AML": "tab:orange"}

# ================================================================ 1) VERİYİ OKU, ETİKETLERİ KONTROL ET
print("\n=== 1) Veri yükleme ve etiket kontrolü ===")
X, C, y, set_bilgisi, etiket = golub_oku()   # X: ifade, C: P/A/M çağrıları, y: ALL/AML
egitim = (set_bilgisi == "Eğitim").values    # hangi hastalar eğitim setinde (True/False)
test = (set_bilgisi == "Test").values
print("Örnek x prob:", X.shape)
print(pd.crosstab(set_bilgisi, y))

# Dosyadaki sütunlar hasta numarası sırasında mı? Değilse etiketleri sırayla vermek hataya yol açar.
print("Test dosyasındaki hasta sırası:", list(X.index[test]))
# Etiketleri sütun sırasına göre verseydim kaç hasta yanlış etiketlenirdi?
yanlis_egitim = (etiket.loc[sorted(X.index[egitim])].values != y[egitim].values).sum()
yanlis_test = (etiket.loc[sorted(X.index[test])].values != y[test].values).sum()
print("Sıraya göre eşleştirseydim yanlış etiket -> eğitim:", yanlis_egitim, " test:", yanlis_test)

# ================================================================ 2) VERİ KALİTESİ
print("\n=== 2) Veri kalitesi ===")
v = X.values
print("Eksik (NaN) sayısı:", np.isnan(v).sum(), " INF sayısı:", np.isinf(v).sum())
print("En küçük / en büyük değer:", v.min(), "/", v.max())

# Negatif ifade değeri olmaz, peki bu negatifler hata mı?
# "A" (absent) çağrısı genin o hastada saptanmadığını gösteriyor. Negatiflerin çoğu "A" ise sorun yok.
negatif = v < 0
print(f"Negatif değer oranı: {negatif.mean():.1%}")
print(f"Negatif değerlerin 'A' olma oranı: {(C.values[negatif] == 'A').mean():.1%}")

# ================================================================ 3) ÖN İŞLEME: kırpma + filtre + log2
print("\n=== 3) Ön işleme ===")
X_kirp = X.clip(100, 16000)                  # çok küçük ve çok büyük değerleri sınırla (Dudoit ve ark., 2002)
en_buyuk, en_kucuk = X_kirp.max(), X_kirp.min()
# hastalar arasında neredeyse hiç değişmeyen genler sınıflandırmaya bilgi katmaz, onları atıyorum
tut = (en_buyuk / en_kucuk > 5) & (en_buyuk - en_kucuk > 500)
L = np.log2(X_kirp.loc[:, tut])              # L: analizlerde kullanacağım log2 veri
print("Filtreden geçen gen sayısı:", tut.sum(), "(literatürde 3571)")

# çarpıklık: 0'a yakınsa dağılım simetrik. Ham veride çok büyük olmasını bekliyorum.
print("Çarpıklık medyanı -> ham:", round(np.median(stats.skew(X.values, axis=1)), 2),
      " log2 sonrası:", round(np.median(stats.skew(L.values, axis=1)), 2))
print(f"Alt sınıra (100) kırpılan değer oranı: {(X_kirp.loc[:, tut] == 100).values.mean():.1%}")

# Şekil: (a) ham değerler, (b) ön işleme sonrası ALL ve AML dağılımı
fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
ax[0].hist(X.values.ravel(), bins=np.arange(-1000, 5001, 75), color="gray")
ax[0].axvline(0, color="black", ls="--")
ax[0].set(title="(a) Ham veri", xlabel="Ham ifade (AD)", ylabel="Ölçüm sayısı")
for s in ("ALL", "AML"):
    d = L[y == s].values.ravel()
    d = d[d > np.log2(100)]                  # alt sınıra kırpılanları çizmiyorum, yoksa tek sütun her şeyi eziyor
    ax[1].hist(d, bins=50, density=True, histtype="step", color=renk[s], label=s)
ax[1].set(title="(b) Ön işleme sonrası", xlabel="log2 ifade", ylabel="Yoğunluk")
ax[1].legend()
fig.savefig(SEKIL / "golub_1_dagilim.pdf")
plt.close(fig)

# ================================================================ 4) BOYUT İNDİRGEME: PCA ve t-SNE (GPU)
print("\n=== 4) PCA ve t-SNE ===")
Z = StandardScaler().fit_transform(L.values.astype(np.float32))   # her gen için z-skor
pca = PCA(n_components=30).fit(Z)
oran = pca.explained_variance_ratio_      # her bileşenin açıkladığı varyans oranı
P = pca.transform(Z)                      # hastaların ilk 30 bileşendeki skorları
print("PC1, PC2, PC3 açıklanan varyans:", oran[:3].round(3))
print("Varyansın yarısı için gereken bileşen sayısı:", np.searchsorted(np.cumsum(oran), 0.5) + 1)

# Bileşenler neyi ayırıyor: hastalığı mı (ALL/AML), yoksa eğitim/test setini mi? (Mann-Whitney testi)
for i in range(5):
    p_sinif = stats.mannwhitneyu(P[y == "ALL", i], P[y == "AML", i]).pvalue
    p_set = stats.mannwhitneyu(P[egitim, i], P[test, i]).pvalue
    print(f"PC{i + 1}: sınıf p = {p_sinif:.2e}   eğitim/test p = {p_set:.2e}")
# Beklemediğim sonuç: PC1 sınıfı değil eğitim/test setini ayırıyor! (6. bölümde ayrıca inceledim)

# t-SNE'yi ilk 30 bileşen üzerinde çalıştırıyorum (gürültüyü azaltıyor). Örnek az olduğu için "exact" yöntemi.
ts = TSNE(n_components=2, perplexity=15, method="exact", random_state=42).fit_transform(P)

# Şekil: (a) PCA, (b) t-SNE. Renk sınıfı, işaret eğitim/test setini gösteriyor.
fig, ax = plt.subplots(1, 2, figsize=(7, 3.3), layout="constrained")
for a, M in ((ax[0], P), (ax[1], ts)):
    for s in ("ALL", "AML"):
        m = (y == s).values
        a.scatter(M[m & egitim, 0], M[m & egitim, 1], marker="o", color=renk[s], label=s + " eğitim")
        a.scatter(M[m & test, 0], M[m & test, 1], marker="^", color=renk[s], label=s + " test")
ax[0].set(title="(a) PCA", xlabel=f"PC1 ({oran[0]:.1%})", ylabel=f"PC2 ({oran[1]:.1%})")
ax[1].set(title="(b) t-SNE", xlabel="t-SNE 1", ylabel="t-SNE 2")
ax[0].legend(fontsize=7)
fig.savefig(SEKIL / "golub_2_pca_tsne.pdf")
plt.close(fig)

# ================================================================ 5) AYIRT EDİCİ GENLER
print("\n=== 5) Ayırt edici genler (ALL ve AML) ===")
all_, aml = L[y == "ALL"], L[y == "AML"]
t, p = stats.ttest_ind(aml, all_, equal_var=False)   # Welch t-testi (varyanslar eşit olmak zorunda değil)
# 3571 test yapıyorum, bu yüzden p değerlerini düzeltmem gerekiyor (Benjamini-Hochberg, FDR)
q = multipletests(p, method="fdr_bh")[1]
fark = (aml.mean() - all_.mean()).values             # log2 fark: 1 = iki kat

sembol = anotasyon_oku(VERI / "golub" / "GPL80.annot.gz")   # prob numarası -> gen sembolü
genler = pd.DataFrame({"prob": L.columns, "sembol": sembol.reindex(L.columns).values,
                       "t": t, "p": p, "q": q, "log2_fark": fark}).sort_values("p")
genler.to_csv(TABLO / "golub_ayirt_edici_genler.csv", index=False)
print("q < 0,05 olan gen:", (q < 0.05).sum(), " | bunlardan farkı iki kattan büyük olan:",
      ((q < 0.05) & (np.abs(fark) > 1)).sum())
print("En ayırt edici 10 gen:")
print(genler.head(10)[["prob", "sembol", "log2_fark", "q"]].to_string(index=False))

# En iyi 30 genle hastaları iki kümeye ayırsam kümeler ALL ve AML ile örtüşür mü?
en_iyi = genler.head(30)
H = L[en_iyi.prob]
Hz = (H - H.mean()) / H.std()
kume = fcluster(linkage(Hz.values, "average", metric="correlation"), 2, "maxclust")
capraz = pd.crosstab(kume, y.values)
print(capraz)
print(f"Küme saflığı: {capraz.max(axis=1).sum() / capraz.values.sum():.1%}")
# DİKKAT: genleri tüm hastaların etiketine bakarak seçtim, bu yüzden bu oran iyimser.

# Şekil: en ayırt edici 30 genin ısı haritası (satır = gen, sütun = hasta, üstteki şerit = sınıf)
H = H.T
H.index = en_iyi.sembol.fillna(en_iyi.prob).values
g = sns.clustermap(H, z_score=0, method="average", metric="correlation", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                   col_colors=y.map(renk).values, xticklabels=False, yticklabels=True, figsize=(7, 7))
g.savefig(SEKIL / "golub_3_isi_haritasi.pdf")
plt.close(g.figure)

# ================================================================ 6) EĞİTİM VE TEST SETLERİ ARASINDAKİ FARK
print("\n=== 6) Eğitim - test farkı (toplu etki var mı?) ===")
# PC1 eğitim ve test setlerini ayırmıştı. Bunun hastalıktan kaynaklanmadığından emin olmak için
# iki seti HER SINIFIN KENDİ İÇİNDE, gen gen karşılaştırıyorum (Kolmogorov-Smirnov testi).
# İki set arasında gerçek fark yoksa genlerin yaklaşık %5'i şans eseri p < 0,05 çıkar.
ks_orani = {}
for s in ("ALL", "AML"):
    A = L[(y == s).values & egitim]
    B = L[(y == s).values & test]
    p_ks = np.array([stats.ks_2samp(A[g], B[g]).pvalue for g in L.columns])
    ks_orani[s] = (p_ks < 0.05).mean() * 100
    print(f"{s}: p < 0,05 olan gen oranı %{ks_orani[s]:.1f} (beklenen %5), "
          f"FDR düzeltmesinden sonra {(multipletests(p_ks, method='fdr_bh')[1] < 0.05).sum()} gen")

# Şekil: (a) PC1 skorları eğitim/test, (b) sınıf içi KS testinde anlamlı gen oranı
fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
ax[0].boxplot([P[egitim, 0], P[test, 0]])
ax[0].set_xticklabels(["Eğitim", "Test"])
ax[0].set(title="(a) PC1 skoru", ylabel="PC1")
ax[1].bar(["ALL", "AML"], [ks_orani["ALL"], ks_orani["AML"]], color=[renk["ALL"], renk["AML"]])
ax[1].axhline(5, color="black", ls="--", label="beklenen (%5)")
ax[1].set(title="(b) Sınıf içi KS testi", ylabel="KS p < 0,05 olan gen (%)")
ax[1].legend()
fig.savefig(SEKIL / "golub_4_egitim_test.pdf")
plt.close(fig)

print("\nBitti. Şekiller ve tablolar 'sonuclar' klasöründe.")
