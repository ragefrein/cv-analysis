# Futsal Vision

Aplikasi lokal untuk analisis rekaman futsal dari satu kamera bergerak. Unggah video, deteksi dan track pemain/bola dengan GPU atau CPU, hubungkan lintasan ke roster, review kandidat passing/penguasaan, catat kejadian lain, dan ekspor statistik dengan bukti video.

## Jalankan di komputer ini

Python virtual environment, model YOLO, dan frontend sudah disiapkan selama implementasi. Dari PowerShell di folder project:

```powershell
.\start.ps1
```

Buka [http://localhost:8000](http://localhost:8000). Hentikan dengan Ctrl+C. Jika PowerShell membatasi script, jalankan interpreter langsung:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Semua data tersimpan di `data/`. Service hanya mendengarkan loopback dan dirancang untuk satu pengguna lokal. Tidak ada cloud upload, akun eksternal, atau API AI berbayar. Aplikasi belum dirancang untuk diekspos langsung ke internet; autentikasi dan worker terpisah diperlukan untuk penggunaan banyak pengguna.

## Setup baru

Persyaratan: Python 3.11–3.12, Node.js 22.12+ atau 24, FFmpeg dan FFprobe di PATH. NVIDIA CUDA opsional. Model kecil dapat berjalan di CPU, dengan waktu proses bergantung perangkat.

```powershell
.\setup.ps1
.\start.ps1
```

Setup membuat `.venv`, memasang dependency, mengunduh YOLO11n resmi (~5.4 MB) ke `models/`, dan build frontend. Unduhan hanya terjadi saat setup; worker analisis tidak mengunduh model atau memasang paket otomatis. Untuk memakai library Python yang memang sudah terpasang, gunakan `setup.ps1 -UseSystemPackages`.

PyTorch CUDA tidak otomatis dijanjikan oleh instalasi pip biasa. Ikuti pemilih instalasi di [pytorch.org](https://pytorch.org/get-started/locally/) jika ingin GPU pada mesin baru. Endpoint `/api/health` menampilkan device yang benar-benar tersedia. Komputer pengembangan ini diuji menggunakan RTX 3050 Ti, PyTorch 2.5.1+cu121, dan YOLO11n.

## Cara penggunaan

1. Buat pertandingan dan tentukan nama/warna jersey kedua tim. Warna dipakai sebagai petunjuk awal; bukan bukti identitas pemain.
2. Upload MP4/MOV/MKV/AVI/WebM/M4V (default maksimum 4 GB). Video divalidasi lalu dibuat proxy H.264 bila diperlukan. Konversi dapat memerlukan waktu setelah transfer selesai. Satu pertandingan menyimpan satu video; buat pertandingan baru untuk rekaman lain.
3. Masukkan roster dan nomor punggung bila tersedia. Kiper dapat ditandai di roster.
4. Jalankan analisis. Pilih sampling 1–15 FPS dan confidence. Sampling rendah lebih ringan tetapi dapat melewatkan aksi singkat.
5. Di review video, lihat kotak deteksi dan track ID. Hubungkan track ke pemain sebenarnya. Track yang berbeda dapat dihubungkan ke pemain sama setelah keluar/masuk frame. Abaikan wasit/official jika terdeteksi sebagai orang.
6. Tinjau kandidat passing dan penguasaan: koreksi waktu, pengirim, penerima, tim, hasil, lalu konfirmasi atau tolak. Tambahkan kejadian yang terlewat.
7. Catat shot, gol, own goal, assist, interception, turnover, recovery, save, block, dribble, foul, kartu, pergantian, kick-in, corner, dan penalti secara manual dengan cuplikan video sebagai bukti.
8. Buka dashboard/statistik dan ekspor CSV pemain, CSV kejadian, JSON lengkap, atau laporan HTML yang dapat dicetak menjadi PDF lewat browser.
9. Finalisasi hanya setelah seluruh kandidat direview. Edit data setelah finalisasi membuka status review kembali.

Tombol simulasi membuat video lapangan, lintasan, dan statistik **sintetis yang diberi label jelas**. Data ini hanya untuk mempelajari UI; bukan contoh akurasi deteksi. Aplikasi awal tidak menciptakan pertandingan palsu otomatis.

## Cara angka dihitung

- Hanya event `confirmed` yang masuk skor, statistik pemain/tim, jaringan passing, dan possession. Kandidat AI selalu `pending` dan tidak langsung menjadi angka final.
- Passing diketahui = sukses + gagal. Akurasi = sukses / passing diketahui. Passing unresolved tetap ditampilkan tetapi tidak diperlakukan sebagai gagal.
- Event gol juga dihitung sebagai shot dan on target. Jika shot yang sama sudah dicatat, tautkan gol ke event shot tersebut; tidak dihitung dua kali. Shot outcome `goal` juga dapat mewakili gol. Own goal menambah skor lawan, bukan gol/shot individual pelaku.
- Penguasaan memerlukan interval awal/akhir. Interval terkonfirmasi tidak boleh bertumpuk. Persentase memakai total durasi interval penguasaan berlabel; waktu lain tetap belum diklasifikasikan, termasuk kemungkinan bola bebas dan jeda.
- Satu nama/nomor roster berbeda dari track ID. Keluar frame bukan substitusi. Time on screen bukan menit bermain.
- `identity_seconds` adalah durasi sampel ketika setidaknya satu track terhubung ke pemain; bukan persentase semua pemain teridentifikasi. Ball-visible coverage adalah estimasi dari sampel, bukan sertifikasi semua sentuhan teramati.
- Riwayat revisi disimpan dalam JSON export. Rerun AI mempertahankan event manual dan event terkonfirmasi/ditolak. Kandidat AI pending serta track mapping diganti oleh run baru; identitas harus ditinjau lagi. Proses gagal/batal mempertahankan hasil run sebelumnya.

## Posisi lapangan dan kamera bergerak

Kalibrasi manual memakai empat titik gambar dan empat titik lapangan berkorespondensi, disimpan untuk interval video tertentu. Koordinat gambar dinormalisasi 0–1; posisi lapangan memakai meter. Default titik lapangan adalah empat sudut lapangan dalam urutan TL, TR, BR, BL.

**Kalibrasi statis ini hanya valid selama sudut/pan/zoom tetap sesuai titik acuan.** Untuk kamera bergerak, gunakan segmen pendek yang benar-benar stabil dan kalibrasi ulang setelah kamera berubah. Belum ada estimasi homography dinamis otomatis. Jangan menganggap satu kalibrasi berlaku sepanjang pertandingan.

Endpoint trajectory default menghasilkan posisi di gambar, bukan heatmap lapangan. `?coordinates=court` hanya memproyeksikan observasi yang termasuk interval kalibrasi manual. Titik di luar lapangan dibuang. Pergerakan di luar kamera tidak diisi; jarak lari total, kecepatan fisik, dan lintasan 3D bola belum dihitung.

## Batas otomatisasi yang sudah nyata

Deteksi memakai model COCO umum, belum dilatih khusus pada footage futsal pengguna. Semua orang dapat termasuk wasit/official. Klasifikasi tim menggunakan heuristik warna torso. BoT-SORT mengompensasi gerak kamera, tetapi identitas bisa putus/tertukar karena occlusion, pan, blur, dan pemain masuk kembali. ReID dan pembacaan nomor punggung otomatis belum diaktifkan.

Kandidat passing/penguasaan menggunakan posisi bola dan pemain sepanjang beberapa sampel. Kedekatan gambar tidak membuktikan kontak kaki. One-touch pass, bola kecil, dan bola di udara bisa gagal. **Shot, gol, assist, pelanggaran, dan statistik lanjut tidak diklaim terdeteksi otomatis.** Sistem menyediakan pencatatan/review yang dapat dipakai untuk melengkapi data.

Akurasi futsal, penghematan waktu analis, dan kualitas pemetaan lapangan belum dapat dinyatakan tanpa rekaman asli dan label pembanding. Demo dan smoke test hanya membuktikan jalur aplikasi/pemrosesan berjalan. Fine-tuning, evaluasi pada pertandingan terpisah, dan kalibrasi dinamis berada pada roadmap di [FUTSAL_ANALYTICS_PLAN.md](FUTSAL_ANALYTICS_PLAN.md).

## Pengembangan dan pengujian

```powershell
# Terminal 1: API
.\start.ps1 -Dev

# Terminal 2: Vite
npm run dev --prefix frontend

# Validasi
.\.venv\Scripts\python.exe -m pytest tests -q
npm run build --prefix frontend

# Smoke end-to-end dengan inferensi GPU/CPU nyata, data terisolasi
.\.venv\Scripts\python.exe scripts/smoke.py
```

Vite di [localhost:5173](http://localhost:5173) memproxy `/api` ke port 8000. Build produksi disajikan dari FastAPI pada port 8000. Jika build pertama dibuat saat API sudah berjalan, restart API agar mount frontend aktif. Dokumentasi endpoint: [localhost:8000/docs](http://localhost:8000/docs).

Pengujian harus memakai direktori sementara di `test-results/`, terpisah dari pertandingan pengguna. Suite mencakup aturan statistik, validasi event/identitas, transaksi/rollback, media range, upload, kalibrasi, dan pipeline. Smoke test inferensi menggunakan fixture sintetis atau aset bawaan Ultralytics dan tidak mengukur akurasi futsal.

Konfigurasi opsional:

| Environment variable | Default |
| --- | --- |
| `FUTSAL_DATA_DIR` | `data/` dalam project |
| `FUTSAL_MODEL` | `models/yolo11n.pt` |
| `FUTSAL_MAX_UPLOAD_MB` | `4096` |

Backup folder `data/` saat server berhenti; JSON export berguna untuk inspeksi dan integrasi, bukan fitur restore otomatis. Penghapusan pertandingan dari UI menghapus data dan media pertandingan tersebut.

## Referensi dan lisensi

Pipeline memakai [Ultralytics tracking](https://docs.ultralytics.com/modes/track), [PyAV](https://pyav.org/docs/stable/), [FastAPI](https://fastapi.tiangolo.com/), dan [Vite](https://vite.dev/). Pendekatan pemisahan tracking/identitas/kalibrasi merujuk [SoccerNet GSR](https://github.com/SoccerNet/sn-gamestate). Bobot, library, serta dataset memiliki ketentuan masing-masing; tinjau lisensi terkait sebelum distribusi atau pemakaian komersial.
#   c v - a n a l y s i s  
 