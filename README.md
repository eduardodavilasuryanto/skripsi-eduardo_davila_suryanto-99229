# Komparasi Performa dan Efisiensi Data Whisper dan XLS-R untuk ASR Bahasa Jawa

Repositori ini berisi kode, konfigurasi, pembagian data, hipotesis per utterance, keluaran analisis, dan log eksperimen untuk skripsi *Komparasi Performa dan Efisiensi Data Model Whisper dan XLS-R untuk ASR Bahasa Jawa Berdasarkan Pemodelan Power Law* (Program Studi Sistem Informasi, Universitas Multimedia Nusantara, 2026). Setiap angka pada BAB III, BAB IV, dan Lampiran B skripsi dapat ditelusuri ke skrip dan berkas keluaran di sini tanpa melatih ulang model.

## Rancangan Penelitian

Penelitian ini membandingkan dua model prapelatihan yang di-*fine-tune* untuk pengenalan wicara bahasa Jawa: Whisper *small* (`openai/whisper-small`, sekitar 244 juta parameter) dan XLS-R 300M (`facebook/wav2vec2-xls-r-300m`, sekitar 315 juta parameter). Keduanya dilatih pada lima tingkat jumlah data training berskala logaritmik, satu kali per kombinasi dengan *seed* 42, sehingga terdapat sepuluh konfigurasi.

Data berasal dari corpus OpenSLR SLR35 (*Large Javanese ASR training data set*), yaitu rekaman *read speech*. Subset kerja terdiri atas 4 dari 16 arsip corpus, berisi 46.309 utterance (74,24 jam). Sesudah 16 utterance disaring karena terlalu pendek atau memuat angka, data dibagi secara *text-disjoint* dengan *seed* 42: *training pool* 41.293 utterance (66,13 jam), validation 1.900 utterance (3,05 jam), dan test 3.100 utterance (5,03 jam). Penutur tidak dipisahkan antarbagian, sehingga evaluasi bersifat *speaker-dependent*.

Kelima tingkat data merupakan subset bersarang dari *training pool*:

| Tingkat | Jam (nominal) | Utterance |
| ------- | ------------: | --------: |
| D1 | 1 | 610 |
| D2 | 3 | 1.862 |
| D3 | 10 | 6.234 |
| D4 | 30 | 18.796 |
| D5 | 66 | 41.293 |

Pelatihan kedua model memakai *batch* efektif 32 utterance, jadwal *learning rate* linear dengan *warmup* 500 langkah sampai langkah 10.000, evaluasi WER validation setiap 250 langkah, dan penghentian dini sesudah empat evaluasi tanpa perbaikan dengan batas bawah 2.000 langkah. *Learning rate* puncak Whisper 1 × 10⁻⁵ dan XLS-R 3 × 10⁻⁴. Data test didekode secara *greedy* dari *checkpoint* dengan WER validation terendah, tanpa model bahasa eksternal. Rincian lengkapnya ada pada berkas `configs/`.

WER dan CER dihitung pada tingkat corpus sesudah normalisasi teks. Selang kepercayaan 95% diperoleh dari 10.000 iterasi *bootstrap* per utterance. Perbedaan kedua model pada setiap tingkat diuji dengan *paired bootstrap* dan koreksi Holm. Hubungan jumlah data dan WER dicocokkan dengan `WER = E₀ + a · D^(−c)`, dengan `E₀` bersama bagi kedua model, dan eksponen `c` dipakai sebagai ukuran efisiensi data.

## Hasil Utama

WER data test (%) dengan selang kepercayaan 95%. Selisih dihitung sebagai WER Whisper dikurangi WER XLS-R, sehingga nilai negatif berarti WER Whisper lebih rendah.

| Tingkat | Whisper *small* | XLS-R 300M | Selisih (poin) |
| ------- | --------------- | ---------- | -------------: |
| D1 (1 jam) | 41,2 [40,2; 42,3] | 47,9 [47,0; 48,7] | −6,6 |
| D2 (3 jam) | 30,6 [29,7; 31,6] | 37,8 [36,9; 38,6] | −7,2 |
| D3 (10 jam) | 22,2 [21,5; 23,0] | 24,0 [23,2; 24,7] | −1,8 |
| D4 (30 jam) | 16,4 [15,8; 17,1] | 17,6 [17,0; 18,3] | −1,2 |
| D5 (66 jam) | 15,4 [14,8; 16,0] | 18,8 [18,1; 19,5] | −3,4 |

Whisper *small* mencapai WER lebih rendah daripada XLS-R 300M pada kelima tingkat, dan selisihnya berbeda dari nol pada setiap tingkat sesudah koreksi Holm. Pada CER, selang selisih D3 dan D4 memuat nol, sehingga keunggulan Whisper pada kedua tingkat itu hanya tampak pada WER.

Eksponen kurva penskalaan Whisper 0,364 [0,337; 0,393] dan XLS-R 0,356 [0,334; 0,378]. Selang selisih keduanya memuat nol, sehingga efisiensi data kedua model tidak terbedakan. Perbedaan kedua model terletak pada tinggi kurva, bukan pada laju penurunannya. Menurut AICc dan prediksi titik yang ditahan, bentuk *power law* tanpa batas bawah justru lebih didukung daripada bentuk dengan `E₀`, dan tiga *run* XLS-R berhenti pada batas 10.000 langkah. Karena itu, parameter kurva dibaca sebagai deskripsi di dalam rentang 1 sampai 66 jam, dan perbandingan eksponennya bersifat eksploratif.

Hasil ini berlaku untuk satu *run* per konfigurasi pada resep pelatihan di atas. Selang kepercayaan hanya mencakup variasi sampel data test, bukan variasi antar-*run*.

## Isi Repositori

```
configs/                      konfigurasi pelatihan D2-D5 dan konfigurasi audit
configs/trial/                konfigurasi D1 (*_d1_trial.yaml), probe VRAM, dan uji determinisme
outputs/splits/               pembagian test, validation, training pool, dan utterance yang disaring
outputs/subsets/              daftar utterance D1-D5
outputs/runs/<run>/           salinan konfigurasi, run_report.json, dan trainer_state.json setiap run
outputs/hypotheses/           hipotesis per utterance kesepuluh konfigurasi pada data test
outputs/results/              metrik, tabel, dan laporan analisis (JSON dan CSV)
logs/                         log setiap eksekusi skrip dan setiap run
requirements.txt              versi terkunci seluruh pustaka
```

| Tahap | Skrip |
| ----- | ----- |
| Lingkungan dan data | `log_environment.py`, `audit_transcript_leakage.py`, `build_text_disjoint_split.py`, `build_nested_subsets.py`, `measure_component_durations.py`, `measure_transcript_characters.py` |
| Pelatihan dan inference | `asr_common.py`, `train_asr.py`, `run_inference.py`, `monitor_run.ps1` |
| Analisis utama | `analysis_common.py`, `bootstrap_metrics.py`, `fit_scaling_curve.py`, `robustness_diacritics.py`, `error_analysis.py`, `summarize_runs.py` |
| Uji sensitivitas | `audit_whisper_token_limit.py`, `audit_normalized_overlap.py`, `sensitivity_defective_test.py`, `analysis_speaker_coverage.py`, `cluster_bootstrap.py`, `curve_model_comparison.py`, `compare_subset_corpus.py`, `data_requirement.py`, `h1_inference_check.py`, `error_boundary.py` |
| Proses pelatihan | `analyze_training_process.py` |
| Audit dan efisiensi | `audit_audio_annotations.py`, `audit_speech_register.py`, `audit_affix_errors.py`, `benchmark_inference.py` |
| Pemeriksa | `verify_references.py`, `verify_paper_numbers.py`, `build_reviewer_package.py` |

*Docstring* di awal setiap skrip menjelaskan tujuan, masukan, dan keluarannya, dan sebagian besar juga memuat perintah untuk menjalankannya.

## Pemetaan ke Skripsi

| Bagian skripsi | Skrip | Keluaran |
| -------------- | ----- | -------- |
| Tabel 3.4 dan 3.5 | `build_text_disjoint_split.py`, `build_nested_subsets.py` | `outputs/build_text_disjoint_split_report.json`, `outputs/build_nested_subsets_report.json` |
| Tabel 3.9 | `measure_component_durations.py` | `outputs/measure_component_durations_report.json` |
| Tabel 3.11 sampai 3.13 | `log_environment.py` | `outputs/environment_report.json` |
| Tabel 4.1 | `summarize_runs.py` | `outputs/results/run_status.csv` |
| Tabel 4.2 dan 4.3 | `bootstrap_metrics.py` | `outputs/results/bootstrap_metrics_*` |
| Tabel 4.4, 4.5, dan Gambar 4.1 | `fit_scaling_curve.py` | `outputs/results/scaling_curve_*` |
| Tabel 4.6 | `robustness_diacritics.py` | `outputs/results/robustness_diacritics*` |
| Tabel 4.7 | `error_analysis.py` | `outputs/results/error_analysis_report.json` |
| Tabel 3.16, Subbab 4.3 sampai 4.5.3 | sepuluh skrip uji sensitivitas | `outputs/results/` dan `outputs/compare_subset_corpus_report.json` |
| Subbab 4.1.2 sampai 4.1.5, Tabel B.1 sampai B.8 | `analyze_training_process.py` | `outputs/results/training_process_*` |
| Subbab 4.1.6, 4.5.4 sampai 4.5.6 | empat skrip audit dan efisiensi | `outputs/results/{audit_audio_annotations_*,speech_register_*,audit_affix_errors_*,benchmark_inference_*}` |

Gambar 4.1, B.1, dan B.2 digambar oleh skrip terpisah di luar repositori ini dari keluaran yang tercantum di atas.

## Menjalankan Ulang

Eksperimen dijalankan pada Windows 11 dengan PowerShell, Python 3.12.10, dan GPU NVIDIA GeForce RTX 4060 Laptop (8 GB). Pustaka dipasang dari versi terkunci:

```powershell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Audio dan metadata corpus tidak termasuk dalam repositori. Arsip `asr_javanese_0` sampai `asr_javanese_3` dapat diunduh dari [OpenSLR SLR35](https://www.openslr.org/35/), lalu diekstrak ke `<akar>\_03_dataset\asr_javanese_<k>\`. Lokasi akar ditetapkan oleh konstanta `SKRIPSI_ROOT` pada `audit_transcript_leakage.py`. Bobot model hasil *fine-tuning* berukuran beberapa gigabita dan tidak disertakan, tetapi tersedia atas permintaan.

Perintah berikut dijalankan dari akar repositori. Opsi `-u` diperlukan agar log terisi selama proses berjalan.

```powershell
# 1. Lingkungan, pembagian data, dan subset (deterministik, seed 42)
python -u log_environment.py 2>&1 | Tee-Object -FilePath logs/log_environment.log
python -u build_text_disjoint_split.py 2>&1 | Tee-Object -FilePath logs/build_text_disjoint_split.log
python -u build_nested_subsets.py 2>&1 | Tee-Object -FilePath logs/build_nested_subsets.log

# 2. Pelatihan dan inference, satu konfigurasi per proses
python -u train_asr.py --config configs/whisper_d2.yaml 2>&1 | Tee-Object -FilePath logs/whisper_d2.log
python -u run_inference.py --config configs/whisper_d2.yaml 2>&1 | Tee-Object -FilePath logs/whisper_d2_inference.log

# 3. Analisis utama, hanya membaca hipotesis tersimpan, tanpa GPU
foreach ($s in 'bootstrap_metrics', 'fit_scaling_curve', 'robustness_diacritics', 'error_analysis', 'summarize_runs') {
    python -u "$s.py" 2>&1 | Tee-Object -FilePath "logs/$s.log"
}
```

Konfigurasi D1 kedua model adalah `configs/trial/whisper_d1_trial.yaml` dan `configs/trial/xlsr_d1_trial.yaml`, sehingga keluarannya bernama `*_d1_trial`. Komentar di kepala kedua berkas itu menyebut run tersebut sebagai percobaan; komentar itu tidak disunting karena SHA-256 berkasnya tercatat pada laporan *run*.

Pembagian data dan subset yang dihasilkan ulang harus identik byte demi byte dengan berkas pada `outputs/splits/` dan `outputs/subsets/`. Analisis pada langkah 3 dan uji sensitivitas bersifat deterministik dengan *seed* 42 dan 10.000 iterasi *bootstrap*. Pelatihan ulang tidak akan menghasilkan angka yang persis sama, karena lintasan mundur *memory-efficient attention* dan CTC pada GPU bersifat nondeterministik. Selisih semacam itu sudah teramati pada uji determinisme di `configs/trial/`. Dua analisis memerlukan bobot model, yaitu `audit_whisper_token_limit.py --regenerate` dan `benchmark_inference.py`, dan `audit_audio_annotations.py` memerlukan audio corpus.

## Catatan Publikasi

Repositori ini dipublikasikan pada 25 November 2026, dimulai dari satu *commit* berisi keadaan akhir repositori kerja penulis. Hash *commit* yang dikutip skripsi sebelum tanggal itu, misalnya `3508dc5` atau `fe9a92b`, merujuk riwayat repositori kerja tersebut, yang tidak dipublikasikan. Isi berkas pada *commit* pertama sama dengan keadaan akhir itu, kecuali README ini, satu kalimat keterangan di awal `monitor_run.ps1`, dan penyuntingan log berikut.

Pada daftar proses CPU teratas di `logs/whisper_d4.monitor.log`, `logs/whisper_d5.monitor.log`, dan `logs/xlsr_d5.monitor.log`, nama satu proses perangkat lunak di luar penelitian diganti `[disunting]` pada lima tempat. Sumber satu catatan manual pada `logs/xlsr_d1_trial.monitor.log` juga diganti `[disunting]`. Persentase CPU-nya dipertahankan. `analyze_training_process.py` hanya membaca total CPU proses Python dari baris tersebut, sehingga sesudah dijalankan ulang seluruh CSV keluarannya identik. Yang berubah hanya SHA-256 keempat log pada `training_process_report.json` beserta waktu eksekusinya.

Log dan laporan lainnya dipertahankan apa adanya, termasuk jalur folder pada komputer penulis.

## Sitasi

```
Suryanto, E. D. (2026). Komparasi performa dan efisiensi data model Whisper dan
XLS-R untuk ASR bahasa Jawa berdasarkan pemodelan power law [Skripsi].
Universitas Multimedia Nusantara.
```
