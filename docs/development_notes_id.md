# Replikasi & Ekstensi — Pereira & Leitão (2026), *Optimizing the Optimizers*

Pipeline Python untuk mereplikasi studi kasus *skylight truss* dari
Pereira, I., & Leitão, A. (2026). Advanced Engineering Informatics, 74, 104690.
https://doi.org/10.1016/j.aei.2026.104690

Data: `data/dataset.csv` dari repositori penulis (CC BY 4.0),
DOI 10.5281/zenodo.18344135 (v2.0). Wajib dikutip di artikel.

## Cara menjalankan

```bash
pip install pymoo scikit-learn scipy pandas matplotlib
cd src
python 01_build_space.py            # surrogate + evaluasi 12.636 desain + Pareto front sebenarnya
python 02_baseline.py               # 4 algoritma x 100 run, Table 1 (CLT), Fig. 2–3
python 03_refpoint_sensitivity.py   # n_runs vs titik referensi HV
python 04_moead_normalization.py    # mengapa MOEA/D-DE gagal
python 05_random_search_refpoint.py # Eksp. 1: 500 konfigurasi x 4 algoritma x 10 run (±10 menit, 2 core)
python 09_rs_analysis.py            # analisis Eksp. 1
python 10_invariant_nruns.py        # aturan n_runs invarian + reliabilitas peringkat
python 06_benchmark_generalization.py  # Eksp. 2: ZDT1/DTLZ2 (±2 menit)
python 07_alt_surrogates.py         # Eksp. 3a: ruang desain RF & RBF
python 08_surrogate_robustness.py   # Eksp. 3b: ulangi analisis inti di tiap surrogate
python 11_figures.py                # gambar final
```
Tahap 1–4 < 1 menit; seluruh pipeline ±15 menit di laptop biasa.

## Pengaturan

| Komponen | Artikel asli | Replikasi ini |
|---|---|---|
| Evaluasi objektif | Frame3DD (±15 s) | Surrogate GBT (log-target) atas seluruh ruang desain |
| Bahasa / pustaka | Julia, Metaheuristics.jl | Python, pymoo 0.6.2 + MOEA/D-DE ditulis ulang mengikuti `MOEAD_DE.jl` |
| Budget | pop 20 × 15 iterasi = 300 evaluasi | sama (diverifikasi: tepat 300) |
| Populasi awal | LHS, sama untuk run ke-i | sama |
| Hyperparameter | default Metaheuristics.jl | sama (Bagian 3.1 artikel) |
| Titik referensi HV | tidak dilaporkan | objektif dinormalisasi ke ideal/nadir front sebenarnya, ref = (1,1; 1,1) |

## Hasil

### Surrogate (90/10 + 5-fold CV) — `results/surrogate_metrics.csv`
| | RMSE | R² uji | R² CV-5 |
|---|---|---|---|
| Displacement | 0,0011 | 0,9998 | 0,9999 |
| Biaya | 107,9 | 0,9994 | 0,9996 |

Setara atau lebih baik daripada model terbaik penulis (GBT displacement RMSE 0,0024; RBF biaya RMSE 107,5).
Pareto front sebenarnya (berbasis surrogate): 251 desain, HV = 1,1783.

### Baseline default — `results/baseline_summary.csv`
| Algoritma | HV rata-rata | SD | % HV sebenarnya | IGD |
|---|---|---|---|---|
| NSGA2 | 1,1620 | 0,0090 | 98,6 | 0,049 |
| SMS-EMOA | 1,1617 | 0,0130 | 98,6 | 0,052 |
| SPEA2 | 1,1575 | 0,0134 | 98,2 | 0,054 |
| MOEA/D-DE | 0,8447 | 0,1845 | 71,7 | 0,303 |

**Pola artikel tereplikasi:** NSGA2 ≈ SMS-EMOA > SPEA2 ≫ MOEA/D-DE; urutan variabilitas
NSGA2 < SMS-EMOA < SPEA2 ≪ MOEA/D-DE sama dengan Table 1; HV MOEA/D-DE turun di bawah populasi
awalnya (Fig. 2 artikel).

### Temuan 1 — Jumlah run dari CLT bergantung pada titik referensi HV
`results/refpoint_sensitivity.csv`, `fig_refpoint_nruns.png`.
Dengan ε = 0,2% dari rata-rata HV dan CI 95%, jumlah run minimum SPEA2 berubah dari
**137 (ref 1,05) → 14 (ref 5) → 1 (ref 20)**, padahal solusinya identik. Peringkat algoritma
hampir tidak berubah. Penulis mengonfirmasi memakai titik referensi terjauh; dengan konvensi itu
(objektif mentah, ref = maksimum ruang desain) replikasi menghasilkan HV ±18.450 (artikel ±20.100)
dan n_runs 6–12, sejalan dengan n_runs = 7 di artikel. Pada ref 1,1 (lazim di literatur),
NSGA2 butuh 58 run, SPEA2 130, SMS-EMOA 120.

#### Eksperimen 1 — Apakah titik referensi mengubah hasil HPO? (`05_*`, `09_*`, `10_*`)
Random search 500 konfigurasi × 10 run per algoritma (ruang Bagian 4.1 artikel), dinilai dengan
ref 1,1 (ternormalisasi) dan titik terjauh (mentah). `rs_refpoint_summary.csv`, `fig_rs_refpoint.png`.

| | τ antar-konvensi | τ split-half (derau) | Konfigurasi terbaik sama? |
|---|---|---|---|
| NSGA2 | 0,92 | 0,69 | ya |
| SPEA2 | 0,93 | 0,60 | ya |
| SMS-EMOA | 0,89 | 0,54 | ya |
| MOEA/D-DE | 0,88 | 0,72 | tidak (selisih 0,6% HV; pilihan ref 1,1 lebih baik di kedua konvensi, p < 0,03) |

**Jawabannya: tidak.** Peringkat konfigurasi praktis identik dan perbedaannya jauh lebih kecil
daripada derau run. Yang berubah hanya *klaim presisi*. Pembingkaian yang tepat untuk naskah:
aturan ε relatif terhadap rata-rata HV memberi rasa presisi palsu, karena luas konstan dari titik
referensi yang jauh menekan CV tanpa menambah daya pembeda antarkonfigurasi.

**Usulan aturan invarian** (`invariant_nruns.csv`): ε = k × SD rata-rata HV antarkonfigurasi.

| | Aturan artikel (ref 1,1 / terjauh) | Usulan k = 0,1 (ref 1,1 / terjauh) |
|---|---|---|
| NSGA2 | 58 / 6 (≈10×) | 10 / 8 |
| SPEA2 | 130 / 11 (≈12×) | 121 / 83 |
| SMS-EMOA | 120 / 12 (≈10×) | 226 / 174 |
| MOEA/D-DE | 45.793 / 7.710 | 224 / 287 |

Kesepakatan peringkat 500 konfigurasi dengan 5 run hanya τ = 0,54–0,72
(`ranking_reliability.csv`): 7 run belum cukup untuk membedakan konfigurasi dengan andal.

#### Eksperimen 2 — Generalisasi pada benchmark (`06_*`)
`bench_nruns_refpoint.csv`, `bench_moead_normalization.csv`, `fig_bench_moead.png`.
ZDT1 dan DTLZ2 (2 objektif), versi skala seimbang dan timpang (f2 × 10⁴), 100 run.
- Temuan 1 berlaku: DTLZ2/NSGA2 butuh 3.079 run pada ref 1,1, 39 pada ref 2, 1 pada ref ≥5.
- Temuan 2 berlaku: pada skala seimbang MOEA/D-DE mentah = ternormalisasi (p = 0,50);
  pada skala timpang runtuh (DTLZ2 300 eval: 12% vs 82%; 1500 eval: 39% vs 93%; ZDT1 1500 eval: 17% vs 52%).
- Catatan: ZDT1 dengan 300 evaluasi belum konvergen untuk semua algoritma (≤18% HV) sehingga tidak informatif.
- Catatan: SPEA2 sedikit terpengaruh skala (density estimation pada ruang objektif mentah).

#### Eksperimen 3 — Ketahanan terhadap surrogate (`07_*`, `08_*`)
`alt_surrogate_metrics.csv`, `surrogate_robustness.csv`. Ruang desain dibangun ulang dengan RF
(R² 0,998/0,993) dan RBF (R² 0,9996/0,99999).

| Surrogate | NSGA2 / SPEA2 / SMS-EMOA (% HV) | n_runs ref 1,1 → terjauh (NSGA2) | MOEA/D-DE mentah → norm |
|---|---|---|---|
| GBT | 98,6 / 98,2 / 98,6 | 58 → 6 | 71,7% → 96,1% |
| RF | 98,8 / 98,5 / 98,5 | 28 → 5 | 64,3% → 96,4% |
| RBF | 98,6 / 98,5 / 98,4 | 41 → 5 | 66,9% → 96,3% |

Semua pola bertahan di ketiga surrogate.

### Temuan 2 — Kegagalan MOEA/D-DE terutama karena skala objektif
`results/moead_normalization_summary.csv`, `fig_moead_normalization.png`.
Fungsi Tchebycheff di Metaheuristics.jl memakai objektif mentah (dicek di `MOEAD_DE.jl`),
padahal displacement ~10⁻² dan biaya ~10³. Hanya dengan menormalisasi objektif:

| MOEA/D-DE | HV rata-rata | % HV sebenarnya |
|---|---|---|
| Tanpa normalisasi (asli) | 0,845 | 71,7 |
| Normalisasi online | 1,122 | 95,2 |
| Normalisasi oracle | 1,133 | 96,1 |

Perbaikan signifikan (Mann-Whitney p < 10⁻³³), tetapi masih di bawah NSGA2 (p < 10⁻²⁷).
Jadi hipotesis penulis (front tidak seragam) hanya menjelaskan sisa selisih kecil;
penyebab utamanya mismatch skala.

## Keterbatasan
- "Kebenaran" berbasis surrogate, bukan Frame3DD; angka HV tidak bisa dibandingkan langsung
  dengan artikel (titik referensi mereka tidak dilaporkan) — yang dibandingkan adalah pola.
- Operator SBX/PM pymoo dan Metaheuristics.jl bisa berbeda dalam detail implementasi.
- HPO hanya random search; GP-BO, TPE, SMAC, irace, dan analisis lanskap disimpan untuk artikel berikutnya.
- σ antarkonfigurasi pada aturan usulan dihitung dari rata-rata 10 run, sehingga masih memuat derau run.
- Koordinat persis titik referensi penulis belum diketahui; angka HV mentah hanya sebanding ordenya.

---

## Tambahan untuk versi Q2

```bash
python 13_extended_benchmarks.py   # DTLZ2 3-obj + 12 problem RE, 6 varian algoritma, 30 run, budget 300 & 3000 (±10 menit)
python 17_ext_summary_fig.py       # ringkasan + Fig. 7
python 15_scale_dose_response.py   # kurva rasio skala DTLZ2 & truss (±5 menit)
python 16_fig6_replot.py           # Fig. 6
python 14_theory_check.py          # verifikasi penurunan HV(R)
```
Suite RE: Tanabe & Ishibuchi (2020), Applied Soft Computing 89:106078, disalin di `third_party/reproblems` (MIT).

### Q2-A — 3 objektif dan problem dunia nyata (`ext_*.csv`, `fig7_extended_benchmarks.png`)
- **Normalisasi MOEA/D-DE (3.000 evaluasi):** lebih baik signifikan pada 11 dari 14 problem, kenaikan
  > 3 poin pada 6 problem (RE21 +57,1; RE36 +26,4; RE25 +19,8; DTLZ2-3 timpang +7,1; RE34 +4,7; RE24 +3,1).
  Sedikit lebih buruk hanya pada 2 problem yang skalanya sudah seimbang (DTLZ2-3 −2,5; RE37 −0,8).
- Rasio rentang objektif saja tidak memprediksi keruntuhan: RE31–RE33 (objektif ketiga = jumlah pelanggaran
  batasan, rentang 10⁷–10⁹) tidak runtuh. Mekanisme tunggal tidak diklaim.
- **Jumlah run aturan A:** berubah > 1,5× pada 25 dari 42 pasangan problem–algoritma antara ref 1,1 dan
  titik terjauh; rasio 0,2× sampai 3.074×. Titik terjauh dari sampel acak setara R = 1,0 pada sebagian objektif
  dan R > 100 pada objektif lain, sehingga arah perubahannya tidak konsisten.
- Catatan: pada 300 evaluasi, normalisasi online kadang lebih buruk (mis. RE23 −9,0) karena estimasi nadir
  dari populasi awal masih kasar.

### Q2-B — Kurva dosis–respons (`dose_response_*.csv`, `fig6_scale_dose_response.png`)
| log₁₀ rasio rentang | −6 | −4 | −2 | 0 | 2 | 4 | 6 |
|---|---|---|---|---|---|---|---|
| DTLZ2, MOEA/D-DE mentah (%) | 35,3 | 37,9 | 39,3 | 92,8 | 40,6 | 36,5 | 34,3 |
| DTLZ2, MOEA/D-DE ternormalisasi (%) | 92,2 | 92,2 | 92,2 | 92,2 | 92,2 | 92,2 | 92,2 |

Pada truss, MOEA/D-DE mentah mencapai 95,9% ketika biaya diskalakan agar seimbang (vs 75,1% pada satuan asli).
NSGA-II, SMS-EMOA, dan versi ternormalisasi datar; SPEA2 hampir datar (turun ke 97,8% pada rasio 10⁶).

### Q2-C — Survei pustaka (`library_survey.csv`)
Tidak satu pun dari 7 pustaka (Metaheuristics.jl, pymoo, PlatEMO, jMetal, jMetalPy, pagmo2, MOEA Framework)
menormalisasi Tchebycheff secara default. PlatEMO (MOEAD type=3) dan jMetal (flag di konstruktor) punya opsi;
semua contoh "default configuration" jMetal memakai `normalizeObjectives = false`.

### Q2-D — Penurunan (`theory_check.csv`)
Untuk R di luar front: HV(A;R) = (R−m₁)(R−m₂) − U(A), dengan m = titik ekstrem dan U tidak bergantung pada R.
Maka CV → SD(m₁+m₂)/R dan n_runs ∝ R⁻². Kemiringan empiris log n vs log R (R ≥ 5): −1,90 / −1,92 / −1,93.
Pada R = 50, peringkat HV ditentukan oleh m₁+m₂ (τ = 0,97–0,99): titik referensi jauh membuat HV
hampir hanya menilai kedua ujung front.

### Q2-E — Validasi rule B (`18_*`, `19_*`, `ruleB_*.csv`, `fig8_ruleB_validation.png`)
```bash
python 18_ruleB_validation_runs.py      # 4 algoritma x 100 konfigurasi x (100 run kebenaran + 50 run pemilihan) = 60.000 run (±25 menit)
python 19_ruleB_validation_analysis.py  # regret, P(top-5), tau; successive halving; konsistensi rule B
```
Kebenaran = rata-rata 100 run (seed 1000–1099); pemilihan = n run dari 50 run lain (seed 2000–2049), 500 ulangan.
Analisis dibatasi n <= 20 karena ulangan dari kumpulan 50 run makin tumpang-tindih.

| Mean regret (poin persen HV front sebenarnya) | n = 1 | n = 3 | n = 7 | n = 20 | SH, budget setara n = 7 |
|---|---|---|---|---|---|
| NSGA-II | 0,158 | 0,109 | 0,074 | 0,053 | 0,031 |
| SPEA2 | 0,202 | 0,191 | 0,165 | 0,136 | 0,136 |
| SMS-EMOA | 0,292 | 0,280 | 0,204 | 0,146 | 0,086 |
| MOEA/D-DE | 1,953 | 0,995 | 0,493 | 0,168 | 0,011 |

Temuan:
- Untuk tiga algoritma GA, regret mutlak kecil (< 0,31 poin) bahkan dengan 1 run: konfigurasi yang
  baik-baik saling berdekatan. Untuk MOEA/D-DE, n kecil benar-benar merugikan (1,95 poin pada n = 1).
- **Rule B konservatif untuk tujuan pemilihan.** n empiris agar E[regret] <= k·σ_b sejati jauh lebih kecil
  daripada prediksi rule B (k = 0,2: NSGA-II 1 vs 3; SPEA2 1 vs 31; SMS-EMOA 7 vs 57; MOEA/D-DE 1 vs 56).
  Rule B menargetkan presisi setiap rata-rata, sedangkan pemilihan hanya perlu memisahkan yang terbaik.
  Jadi rule B tetap invarian terhadap titik referensi, tetapi berfungsi sebagai batas atas yang aman, bukan nilai efisien.
- **Racing lebih efisien:** successive halving dengan total budget sama menurunkan regret 1,2–45× dibanding
  n tetap (mis. SMS-EMOA 0,086 vs 0,204; NSGA-II 0,031 vs 0,074; MOEA/D-DE 0,011 vs 0,493 pada budget n = 7),
  dengan selisih terkecil pada SPEA2 (0,136 vs 0,165, sekitar 1,2×).
- Dengan n = 7 (artikel asli), τ peringkat 100 konfigurasi = 0,69–0,81 dan P(terpilih top-5) = 0,23–0,83.

### Q2-F — Perbandingan metode HPO (`20_*`, `21_*`, `hpo_*.csv`, `fig5_hpo_methods.png`)
```bash
pip install optuna smac scikit-optimize
python 20_hpo_comparison.py   # 5 metode x 4 algoritma x sesi, budget 500 run; bisa dilanjutkan (resumable)
python 21_hpo_analysis.py     # hanya memakai ulangan yang lengkap
```
Lima sesi lengkap per kombinasi (sisanya dihentikan karena batas waktu komputasi).
GP-BO memakai scikit-optimize (GP + EI) karena GPSampler Optuna membutuhkan PyTorch.

| Mean regret (poin persen) | NSGA-II | SPEA2 | SMS-EMOA | MOEA/D-DE | Peringkat rata-rata |
|---|---|---|---|---|---|
| Random search | 0,306 | 0,218 | 0,231 | 0,701 | 3,75 |
| TPE | 0,114 | 0,247 | 0,188 | 0,297 | 2,75 |
| GP-BO | 0,022 | 0,082 | 0,152 | 0,220 | 1,25 |
| SMAC | 0,073 | 0,053 | 0,265 | 1,251 (1 sesi pencilan; median 0,31) | 3,00 |
| Racing (RS+SH) | 0,230 | 0,282 | 0,292 | 0,353 | 4,25 |

Metode berbasis model lebih baik daripada RS pada 9 dari 12 pasangan; signifikan hanya pada 3
(GP-BO dan SMAC pada NSGA-II, SMAC pada SPEA2). RS+SH memakai 442 run (pembulatan) dan tidak lebih
baik daripada RS sebagai strategi HPO tersendiri.

### Q2-G — Penurunan untuk M objektif (`22_theory_M_objectives.py`, `theory_M3_*.csv`)
Proposisi (bukti di Lampiran A naskah): HV(A;R) = R^M − R^(M−1)·Σ m_k + O(R^(M−2)).
- Ekstrem berbeda antar-run (7 kasus 3-objektif): kemiringan log n vs log R = −1,73 … −1,99; τ(HV, −Σm) pada R=50 = 0,92–1,00.
- Ekstrem identik di semua run (5 kasus): kemiringan −2,98 … −4,93 (prediksi ≤ −4 untuk orde kedua).
- Sisa (∏(R−m_k) − HV)/R^(M−2) tetap terbatas (0,01–0,71).
