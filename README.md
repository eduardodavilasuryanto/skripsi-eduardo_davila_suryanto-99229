# Kode Penelitian Skripsi

Kode untuk skripsi **"Analisis Perbandingan Whisper dan XLS-R terhadap Performa Automatic Speech Recognition Bahasa Jawa pada Variasi Jumlah Data Training"**.

Berkas ini adalah ringkasan operasional. **Dokumen konvensi lengkap** berada di:

```
..\_02_context-MD-files\strutkur-script-python.md
```

Dokumen tersebut memuat delapan bagian: keputusan pokok, empat alasan pemilihan skrip, pemetaan per milestone, susunan folder, sepuluh butir konvensi penulisan, perintah PowerShell, pengecualian, dan daftar periksa sebelum angka masuk naskah. Baca dokumen itu sebelum menulis skrip baru.

> **Catatan repositori publik:** repositori ini dipublikasikan pada 25 November 2026 sebagai paket pemeriksa, dimulai dari satu *commit* berisi keadaan akhir repositori kerja penulis. Hash *commit* yang dikutip naskah sebelum tanggal itu, misalnya `3508dc5` atau `fe9a92b`, merujuk riwayat repositori kerja tersebut, yang tidak dipublikasikan. Isi berkas pada *commit* pertama sama dengan keadaan akhir itu, kecuali README ini dan penyuntingan log yang dijelaskan pada bagian Analisis Proses Pelaksanaan. Bobot model, *checkpoint*, dan audio corpus tidak termasuk.

> **Catatan riwayat:** spesifikasi sebelumnya disimpan pada `python-scripts-location.md` di folder ini dan hilang sebelum 17 September 2026. Isinya sudah ditulis ulang pada `strutkur-script-python.md`, dan berkas README ini ditempatkan di folder yang sama dengan kode agar konvensinya dapat ditemukan dari tempat kode berada, sekaligus terlindungi riwayat git repositori ini.

---

## Aturan Pokok

Seluruh kode yang menghasilkan angka atau berkas yang masuk ke naskah ditulis sebagai **skrip Python yang dijalankan dari PowerShell**.

> Notebook tidak menghitung, notebook hanya menggambar.

Notebook analisis diawali dengan membaca berkas hasil yang sudah diproduksi skrip, bukan dengan memuat model atau menjalankan bootstrap. Empat alasannya, yaitu pengelolaan VRAM, urutan import CUDA, reproduktibilitas saat sidang, dan durasi pelatihan, diuraikan pada Bagian 2 dokumen konvensi. Pada pelaksanaannya notebook tidak dipakai sama sekali: seluruh gambar naskah dibuat skrip `render_*.py` pada `..\_06_dokumen-skripsi\_02_figure-generation\`, yang folder dan titik pulihnya di luar repositori ini.

---

## Susunan Folder

```
_01_python-scripts/
├── README.md                          (berkas ini)
├── requirements.txt                   (versi terkunci seluruh lingkungan)
├── configs/
│   ├── whisper_d2.yaml ... whisper_d5.yaml
│   ├── xlsr_d2.yaml    ... xlsr_d5.yaml
│   └── trial/                         (run percobaan dan probe; *_d1_trial.yaml = konfigurasi D1)
│
│   Persiapan data dan lingkungan
├── log_environment.py                 (pencatat lingkungan, Tabel 3.11 sampai 3.13)
├── audit_transcript_leakage.py        (kebocoran transkrip dan statistik corpus)
├── build_text_disjoint_split.py       (penyaringan dan pembagian text-disjoint, seed 42)
├── build_nested_subsets.py            (subset bersarang D1 sampai D5, seed 42)
├── measure_component_durations.py     (durasi komponen graf penutur dan transkrip)
├── measure_transcript_characters.py   (statistik karakter transkrip corpus penuh)
├── verify_references.py               (pemeriksa metadata rujukan)
│
│   Pelatihan dan inference
├── asr_common.py                      (normalisasi teks, metrik, dataset bersama)
├── train_asr.py
├── run_inference.py
├── monitor_run.ps1                    (pemantau run pelatihan, hanya membaca)
│
│   Analisis Milestone 3
├── analysis_common.py                 (modul bersama analisis)
├── bootstrap_metrics.py               (WER dan CER berselang, uji H1)
├── fit_scaling_curve.py               (kurva penskalaan, uji H2 dan H3)
├── robustness_diacritics.py           (uji sensitivitas normalisasi diakritik)
├── error_analysis.py                  (bahan analisis kesalahan D1 dan D5)
├── summarize_runs.py                  (rekap status pelatihan, Tabel 4.1)
│
│   Analisis tambahan Revisi 01
├── audit_whisper_token_limit.py       (batas 128 token Whisper)
├── audit_normalized_overlap.py        (transkrip sama antarbagian)
├── sensitivity_defective_test.py      (tanpa dua utterance test cacat)
├── analysis_speaker_coverage.py       (penutur dikenal dan belum dikenal)
├── cluster_bootstrap.py               (bootstrap per penutur dan per prompt)
├── curve_model_comparison.py          (AIC, AICc, BIC, leave-one-out)
├── compare_subset_corpus.py           (subset kerja terhadap corpus penuh)
├── data_requirement.py                (jam data bagi WER sasaran)
├── h1_inference_check.py              (Holm, Bonferroni, permutasi H1)
├── error_boundary.py                  (galat batas kata)
├── build_reviewer_package.py          (paket pemeriksa ZIP dan manifest)
│
│   Analisis proses pelaksanaan
├── analyze_training_process.py        (lama, laju, dinamika, GPU, inference; Lampiran B)
│
│   Analisis tambahan revisi pembimbing
├── audit_audio_annotations.py         (format audio, indikator sinyal, penapisan anotasi)
├── audit_speech_register.py           (sampel dan skor eksploratif tingkat tutur)
├── audit_affix_errors.py              (operasi karakter, digraf, dan afiks)
├── benchmark_inference.py             (RTF dan memori GPU batch 1)
│
│   Pemeriksa paper
├── verify_paper_numbers.py            (angka dan rujukan paper)
│
├── outputs/
│   ├── splits/                        (test, validation, train_pool, excluded; jangan disunting)
│   ├── subsets/                       (d1.tsv sampai d5.tsv; jangan disunting)
│   ├── runs/                          (salinan konfigurasi dan laporan setiap run)
│   ├── models/                        (bobot model, tidak masuk git)
│   ├── hypotheses/                    (hipotesis per utterance, 10 konfigurasi)
│   └── results/                       (JSON dan CSV hasil analisis)
└── logs/
```

Konfigurasi dipisah sebagai berkas YAML agar perbedaan antarkonfigurasi satu model dapat dibuktikan di depan penguji hanya terletak pada `run_name` dan `train_subset`, sesuai tuntutan variabel terkendali pada Subbab 8.1 dokumen penelitian.

**Konfigurasi D1.** Sejak 5 Oktober 2026 hasil D1 kedua model berasal dari run percobaan Milestone 1, yaitu `configs/trial/whisper_d1_trial.yaml` dan `configs/trial/xlsr_d1_trial.yaml`; berkas `configs/whisper_d1.yaml` dan `configs/xlsr_d1.yaml` tidak dibuat. Kedua berkas percobaan memuat nilai grid, sehingga konfigurasi D2 sampai D5 disalin dari nilainya dan hanya berbeda pada `run_name` dan `train_subset`. Komentar kepalanya, yang menyatakan run tersebut tidak dipakai sebagai hasil Milestone 2 dan D1 dijalankan ulang, sudah tidak berlaku; berkasnya sengaja tidak disunting karena SHA-256-nya tercatat pada laporan run dan log. Keluaran D1 tetap bernama `*_d1_trial` pada `outputs/runs/`, `outputs/hypotheses/`, dan `outputs/results/`. Pemakaian ini sah selama `train_asr.py`, `run_inference.py`, dan `asr_common.py` tidak berubah dari *commit* `5fda5f9` yang menjalankan kedua run; perubahan yang menyentuh perilakunya mewajibkan D1 dijalankan ulang.

---

## Sepuluh Butir Konvensi Penulisan Skrip

1. Nama berkas `snake_case`, deskriptif, tanpa singkatan yang tidak lazim.
2. *Module docstring* di bagian atas memuat tujuan, masukan, keluaran, dan perintah untuk menjalankan.
3. Import dikelompokkan: pustaka standar, pustaka pihak ketiga, lalu modul lokal.
4. Konstanta diletakkan di bagian atas berkas setelah import, bukan tersebar di dalam fungsi.
5. *Type hints* pada seluruh signature fungsi.
6. Guard `if __name__ == "__main__":` pada setiap skrip yang dapat dieksekusi.
7. Mengikuti PEP 8.
8. `encoding="utf-8"` dinyatakan eksplisit pada setiap operasi baca dan tulis berkas. **Wajib**, karena transkrip bahasa Jawa memuat karakter beraksen dan Windows tidak memakai UTF-8 sebagai default.
9. Tidak ada kredensial yang ditulis langsung di dalam kode.
10. *Random seed* ditetapkan eksplisit dan dicatat ke log pada setiap eksekusi.

---

## Cara Menjalankan

Aktifkan `skripsi-venv` lebih dulu. Jangan menjalankan pelatihan GPU melalui `Start-Job`.

Satu konfigurasi dengan pencatatan log:

```powershell
python -u train_asr.py --config configs/whisper_d2.yaml 2>&1 |
    Tee-Object -FilePath logs/whisper_d2.log
```

Seluruh konfigurasi satu model secara berurutan, yaitu D2 sampai D5, karena D1 berasal dari `configs/trial/`:

```powershell
foreach ($cfg in Get-ChildItem configs/whisper_d*.yaml) {
    python -u train_asr.py --config $cfg.FullName 2>&1 |
        Tee-Object -FilePath "logs/$($cfg.BaseName).log"
}
```

Flag `-u` **wajib**. Tanpa flag tersebut Python menahan keluaran di buffer dan log baru terisi setelah proses selesai, sehingga pelatihan yang berlangsung berjam-jam tidak dapat dipantau. Setiap iterasi `foreach` menjalankan proses Python baru, sehingga VRAM dibebaskan sepenuhnya di antara konfigurasi.

### Analisis Milestone 3

Keempat skrip analisis membaca berkas hipotesis kesepuluh konfigurasi, tidak memakai GPU, dan masing-masing selesai dalam hitungan detik sampai satu menit:

```powershell
foreach ($s in 'bootstrap_metrics', 'fit_scaling_curve', 'robustness_diacritics', 'error_analysis') {
    python -u "$s.py" 2>&1 | Tee-Object -FilePath "logs/$s.log"
}
```

Seluruhnya memakai seed 42 dan 10.000 iterasi *bootstrap* yang identik antarskrip. Rincian keputusan analisisnya, yaitu jam data terukur, selang persentil, ambang AIC 2, dan Holm atas nilai *p*, tercatat pada docstring masing-masing skrip. Argumen `--n-bootstrap` dan `--results-dir` hanya untuk uji coba dan tidak dipakai bagi angka naskah. Teks yang dicetak skrip dibatasi ASCII, karena keluaran Python yang disalurkan ke PowerShell memakai halaman kode cp1252.

### Rekap Status Pelatihan

`summarize_runs.py` merangkum `run_report.json` dan `trainer_state.json` kesepuluh konfigurasi menjadi `outputs/results/run_status.csv` dan `run_status_report.json`, sumber Tabel 4.1 `bab-4.md`. Skrip ini hanya membaca, tidak memakai GPU, deterministik tanpa *seed*, dan berhenti sebelum menulis keluaran bila nilai pada kedua berkas atau SHA-256 salinan konfigurasi tidak cocok:

```powershell
python -u summarize_runs.py 2>&1 | Tee-Object -FilePath logs/summarize_runs.log
```

### Analisis Tambahan Revisi 01

Sebelas skrip menjawab masukan Revisi 01 tanpa training baru. Seluruhnya membaca keluaran yang sudah ada, memakai seed 42 dan 10.000 iterasi *bootstrap* yang sama dengan analisis utama bila melakukan *bootstrap*, dan hanya `audit_whisper_token_limit.py --regenerate` yang memakai GPU untuk dekode ulang. Skrip, log setiap eksekusinya pada `logs/`, dan laporan JSON dan CSV-nya pada `outputs/` dan `outputs/results/` di-commit. Laporan ikut di-commit karena tidak semuanya dapat dihasilkan ulang dari isi repositori: `audit_whisper_token_limit.py --regenerate` membutuhkan bobot model pada `outputs/models/`, yang tidak masuk git, dan `compare_subset_corpus.py` membaca metadata corpus SLR35 lokal. Pada mesin penelitian, perintah berikut menghasilkannya ulang:

```powershell
python -u audit_whisper_token_limit.py --regenerate 2>&1 |
    Tee-Object -FilePath logs/audit_whisper_token_limit.log
foreach ($s in 'audit_normalized_overlap', 'sensitivity_defective_test',
               'analysis_speaker_coverage', 'cluster_bootstrap',
               'curve_model_comparison', 'compare_subset_corpus',
               'data_requirement', 'h1_inference_check', 'error_boundary') {
    python -u "$s.py" 2>&1 | Tee-Object -FilePath "logs/$s.log"
}
```

`build_reviewer_package.py` mengemas seluruh berkas terlacak dan belum terlacak yang tidak diabaikan menjadi arsip ZIP beserta manifest SHA-256. Sejak 25 November 2026 paket pemeriksa adalah repositori publik ini sendiri, sehingga skrip ini tidak dijalankan lagi dan pembaruan dilakukan lewat *commit*. Arsip terakhirnya, `paket-pemeriksa-20261125.zip`, berisi keadaan repositori sebelum dipublikasikan.

### Analisis Proses Pelaksanaan

`analyze_training_process.py` menganalisis jalannya pelatihan dan inference kesepuluh konfigurasi, yaitu lama dan rincian waktu setiap *run*, laju per 25 langkah dan periode lambat, dinamika loss, norma gradien, dan WER validation, memori GPU, suhu, daya, dan pembatasan clock GPU, beban CPU dan disk, peristiwa charger, inference, serta keseragaman lingkungan antarrun. Skrip ini hanya membaca `outputs/runs/<run>/run_report.json` dan `trainer_state.json`, `logs/<run>.log`, `logs/<run>.monitor.log` tulisan `monitor_run.ps1`, `logs/kernel_power_105_20261004-20261006.log`, dan `outputs/results/<run>_test_metrics.json`. Skrip tidak memakai GPU, deterministik tanpa *seed*, dan berhenti sebelum menulis keluaran bila jumlah langkah terukur yang direkonstruksi dari log, waktu langkahnya, atau lama evaluasinya tidak cocok dengan laporan *run*. Keluarannya pada `outputs/results/`: `training_process_report.json`, empat CSV per *run*, per jendela laju, per evaluasi, dan per potret pemantau, serta delapan CSV `training_process_appendix_b*.csv` yang memuat nilai Tabel B.1 sampai B.8 `lampiran.md` dalam satuan naskah. Kedua gambar Lampiran B digambar `render_dinamika_pelatihan.py` dan `render_proses_run.py` pada `..\_06_dokumen-skripsi\_02_figure-generation\` dari keluaran ini, sehingga keduanya dirender ulang bila skrip ini dijalankan ulang.

```powershell
python -u analyze_training_process.py 2>&1 | Tee-Object -FilePath logs/analyze_training_process.log
```

Sebelum repositori dipublikasikan pada 25 November 2026, lima kemunculan nama satu proses perangkat lunak di luar penelitian pada daftar proses CPU teratas di `logs/whisper_d4.monitor.log`, `logs/whisper_d5.monitor.log`, dan `logs/xlsr_d5.monitor.log`, serta sumber satu catatan manual pada `logs/xlsr_d1_trial.monitor.log`, diganti `[disunting]`. Persentase CPU-nya dipertahankan. Skrip ini hanya membaca total CPU proses python dari baris tersebut, sehingga sesudah dijalankan ulang seluruh CSV keluarannya identik; yang berubah hanya SHA-256 keempat log pada `training_process_report.json` beserta waktu eksekusinya.

### Analisis Tambahan Revisi Pembimbing

Empat skrip menanggapi masukan revisi tanpa pelatihan ulang dan tanpa mengubah pembagian, hipotesis tersimpan, atau hasil utama. Konfigurasi masing-masing berada pada `configs/`, log pada `logs/`, dan keluaran pada `outputs/results/`. Keluaran ikut di-commit karena sebagian membaca audio lokal, label yang disimpan, atau *checkpoint* pada `outputs/models/`, yang tidak masuk git.

- `audit_audio_annotations.py` memeriksa format dan durasi seluruh berkas subset kerja, menghitung indikator energi bingkai rendah, menapis indikasi masalah anotasi, membuat daftar pemeriksaan dengar, dan menghitung WER deskriptif dari hipotesis tersimpan. Indikator sinyal bukan SNR, dan penanda otomatis bukan konfirmasi cacat anotasi.
- `audit_speech_register.py` memilih sampel data test dengan *seed* 42 tanpa membaca hipotesis, lalu menghitung WER dan CER dari label tingkat tutur pada `configs/speech_register_labels.csv`. Label tersebut berasal dari transkrip rujukan dan belum diverifikasi lewat audio.
- `audit_affix_errors.py` mencocokkan operasi karakter dengan CER utama, menjajarkan digraf `ng`, `ny`, `dh`, dan `th`, serta mengaudit afiks pada daftar bentuk kata tersegmentasi. Skrip ini tidak menghitung *Phoneme Error Rate*.
- `benchmark_inference.py` mengukur RTF dan puncak memori PyTorch kedua *checkpoint* D5 dengan *batch* 1 pada sampel data test yang sama. Skrip ini memerlukan GPU dan model lokal, serta tidak mengukur persentil latensi per permintaan atau seluruh memori proses GPU.

```powershell
python -u audit_audio_annotations.py 2>&1 | Tee-Object -FilePath logs/audit_audio_annotations.log
python -u audit_speech_register.py sample
python -u audit_speech_register.py score 2>&1 | Tee-Object -FilePath logs/audit_speech_register.log
python -u audit_affix_errors.py 2>&1 | Tee-Object -FilePath logs/audit_affix_errors.log
python -u benchmark_inference.py whisper_d5 2>&1 |
    Tee-Object -FilePath logs/benchmark_inference_whisper_d5.log
python -u benchmark_inference.py xlsr_d5 2>&1 |
    Tee-Object -FilePath logs/benchmark_inference_xlsr_d5.log
```

Tahap `sample` pada `audit_speech_register.py` hanya menulis ulang daftar sampel. Jangan mengubah `configs/speech_register_labels.csv` tanpa memeriksa ulang setiap label, karena tahap `score` memeriksa kecocokan nomor dan ID sampel sebelum menghitung metrik.

### Statistik Transkrip dan Pemeriksa Paper

`measure_transcript_characters.py` menghasilkan statistik karakter, ejaan, dan panjang transkrip corpus penuh yang dikutip BAB III Subbab 3.1 dan 3.3.4. `verify_paper_numbers.py` mencocokkan angka, tabel, abstrak, dan daftar pustaka paper pada `..\_06_dokumen-skripsi\_04_paper\paper.md` dengan keluaran skrip dan registri rujukan. Keduanya hanya membaca, deterministik tanpa *seed*, dan tidak memakai GPU. Laporan `measure_transcript_characters.py` di-commit, karena masukannya, metadata corpus SLR35, berada di luar repositori. Laporan `verify_paper_numbers.py` juga di-commit sebagai potret pemeriksaan 9 November 2026; isinya hanya berlaku bagi `paper.md` dengan SHA-256 yang tercatat di dalamnya, karena paper berada di luar repositori.

```powershell
python -u measure_transcript_characters.py 2>&1 |
    Tee-Object -FilePath logs/measure_transcript_characters.log
python -u verify_paper_numbers.py 2>&1 | Tee-Object -FilePath logs/verify_paper_numbers.log
```

---

## Daftar Periksa Sebelum Angka Masuk Naskah

Sebuah angka layak dikutip di skripsi apabila keempat hal berikut terpenuhi:

1. Dihasilkan oleh skrip, bukan oleh sel notebook.
2. Terdapat berkas log yang mencatat waktu eksekusi, konfigurasi, dan *random seed*.
3. Berkas konfigurasi yang dipakai tersimpan dan tidak diubah setelah eksekusi.
4. Perintah untuk menghasilkan ulang angka tersebut dapat dituliskan dalam satu baris.

---

## Prasyarat yang Mudah Terlewat

Pada Milestone 2, **simpan hipotesis per-utterance** berupa pasangan `id`, referensi ternormalisasi, dan hipotesis ternormalisasi untuk seluruh 10 konfigurasi ke `outputs/hypotheses/`. Berkas ini merupakan prasyarat bagi *paired bootstrap* pada H1 dan bagi *bootstrap* parameter kurva pada Subbab 8.0c. Menyimpannya sejak awal tidak menambah biaya apa pun, sedangkan melewatkannya membuat seluruh analisis Milestone 3 memerlukan inference ulang atas sepuluh model. Prasyarat ini sudah dipenuhi: kesepuluh berkas `<run_name>_test.tsv` ada di `outputs/hypotheses/` dan masuk git sejak commit `2de9a7b`.
