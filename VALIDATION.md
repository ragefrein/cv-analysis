# Hasil verifikasi — 31 Oktober 2026

## Status aplikasi

Aplikasi lokal berjalan pada http://localhost:8000, dengan frontend produksi disajikan langsung oleh FastAPI. Jalankan ulang menggunakan `start.ps1`. Tidak ada deployment publik atau upload rekaman ke layanan eksternal.

## Verifikasi yang selesai

- `python -m pytest tests -q --tb=short`: **62 passed** dalam 9.27 detik pada run akhir. Dua peringatan deprecation berasal dari adapter TestClient Starlette/httpx dan alias AnyIO; tidak ada kegagalan tes.
- `npm run build` di `frontend/`: TypeScript dan build Vite berhasil. Build terakhir `index-BqN83psg.js`, 315.05 kB (93.51 kB gzip).
- `scripts/smoke.py`: pengujian terisolasi memakai aset gambar bawaan Ultralytics yang dibuat menjadi rekaman dua detik. Upload, validasi metadata, streaming byte-range, inferensi YOLO/BoT-SORT pada `cuda:0`, empat track, pemetaan roster, kejadian, statistik, seluruh format ekspor, dan proyeksi kalibrasi berhasil. Waktu inferensi fixture pada run tersebut sekitar 5.3 detik; ini bukan benchmark pertandingan futsal.
- Browser: empty state dan layout mobile/desktop, pembuatan demo sintetis, video yang dapat diputar beserta overlay, tambah passing manual dengan pengirim/penerima, konfirmasi kandidat dan perubahan statistik, laporan cetak, empat titik kalibrasi dan penyimpanannya, serta kontrol waktu kalibrasi sudah diperiksa.
- Browser: kandidat passing manual tidak masuk hitungan sebelum dikonfirmasi. Setelah konfirmasi, total passing demo berubah dari 6 ke 7 dan akurasi dari sekitar 83% ke 86%, sesuai enam passing sukses dari tujuh hasil diketahui.

Screenshot hasil di `test-results/dashboard-preview.jpg` menampilkan data simulasi yang dilabeli jelas. Pertandingan demo sekarang memuat contoh review browser dan segmen kalibrasi manual 0–5 detik; semuanya tetap data sintetis.

## Batas yang masih berlaku

Detektor adalah model COCO umum, bukan model yang sudah dilatih pada pertandingan futsal pengguna. AI memberikan deteksi/tracking dan kandidat passing/penguasaan; shot, gol, assist, foul, kartu, serta aksi lain dilengkapi melalui anotasi/review manual. Identitas roster perlu ditinjau, dan official dapat dikecualikan.

Kalibrasi bersifat manual pada segmen stabil. Belum ada kalibrasi dinamis otomatis, identitas berbasis OCR nomor punggung, evaluasi akurasi pada pertandingan futsal asli, atau statistik jarak/kecepatan total, xG, dan pressing. Rekaman di luar frame tidak direkonstruksi sebagai fakta.

Pengujian infrastruktur yang lolos tidak membuktikan akurasi statistik otomatis pada footage futsal. Tahap berikutnya untuk meningkatkan otomatisasi adalah mengumpulkan rekaman asli dan anotasi pembanding, kemudian melatih/evaluasi model dan event engine terhadap data itu.
