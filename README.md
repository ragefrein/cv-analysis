# Futsal Vision

Aplikasi desktop Python/PySide6 untuk analisis video futsal: dua tim dengan warna box berbeda, skeleton tubuh, identitas pemain yang disimpan per video, dan papan taktik 2D.

Fokus saat ini adalah **tagging identitas yang dapat diverifikasi**. Video acuan utama: `1031(2).mp4` (1920×1080, 30 fps, sekitar 2:07). Aplikasi membuka file itu saat dijalankan tanpa argumen; file lain dapat diberikan lewat `python app.py "video.mp4"`.

## Menjalankan

```powershell
pip install -r requirements.txt
python download_models.py
python app.py
```

Model pose sudah tersedia pada workspace ini. `download_models.py` hanya mengunduh jika file belum ada. NVIDIA/CUDA digunakan jika tersedia; komputer tanpa CUDA menggunakan CPU dan membutuhkan waktu lebih lama.

## Cara memakai

1. **Buka Video**, lalu **Analisis**. Analisis berjalan offline; hasil tampil setelah selesai. Video di atas 3 menit dapat dianalisis seluruhnya atau hanya 2 menit pertama.
2. Warna biru menunjukkan **Tim A**, oranye **Tim B**, dan abu-abu **Belum pasti**. Kelompok tim diperkirakan dari warna jersey; kiper dengan jersey berbeda dapat memerlukan penetapan tim manual.
3. **Tampilkan skeleton tubuh** mengaktifkan garis bahu, lengan, pinggul, dan kaki dari model pose. Titik tubuh yang tidak cukup yakin tidak digambar.
4. Label `P0 ?` berarti ID sementara. Pilih pemain, tentukan tim dan nomor punggung bila terbaca, lalu klik **Simpan & verifikasi pemain**. Nama tetap bisa diberikan. Wasit dan bukan pemain dapat ditandai **abaikan**, sehingga tidak muncul di box, hitungan aktif, dan papan taktik. Nomor yang sama pada tim sama tidak bisa diverifikasi untuk dua identitas terpisah.
5. **Tinjau tagging** menampilkan dugaan pasangan fragmen dengan foto dan waktu kemunculan. Pilih **Orang sama · gabungkan**, **Orang berbeda**, atau **Buka frame sumber** untuk memeriksa videonya. Skor jarak penampilan bukan probabilitas atau persentase akurasi. Penggabungan ditolak bila kedua ID pernah terlihat bersamaan atau nomor punggungnya berbeda. **Pisahkan ID mulai frame ini** memperbaiki satu ID yang terlanjur berpindah ke orang lain; bagian sebelumnya mempertahankan nama/nomornya.
6. Papan taktik mengikuti titik kaki pemain pada frame yang diputar, dijeda, atau digeser. **Kalibrasi 4 sudut** memungkinkan pemetaan perspektif ke lapangan penuh: pilih kiri atas, kanan atas, kanan bawah, lalu kiri bawah pada frame yang memperlihatkan seluruh lapangan. Klik kanan membatalkan titik. Kalibrasi mengikuti estimasi gerak kamera dalam shot yang sama. **Gunakan posisi otomatis** menghapus kalibrasi shot tersebut.
7. **Koreksi batas area bermain** menentukan 3–12 titik poligon area bermain yang terlihat; perimeter merah, bangku, dan tribun harus berada di luar. Koreksi langsung menghapus deteksi luar area pada shot tersebut dan disimpan untuk analisis ulang. Pemulihan pemain yang sebelumnya tidak terdeteksi membutuhkan analisis ulang. Ini berbeda dari kalibrasi papan taktik.

`python audit_reference.py` memeriksa 30 lokasi dalam/luar yang ditandai manual pada empat frame video acuan, termasuk gambar dekat tanpa lapangan. Laporan dan gambar pemeriksaan tersimpan di `outputs/reference/`. Pemeriksaan titik ini menguji penyaringan area, bukan akurasi identitas pemain sepanjang video.

Hasil lengkap, foto kecil pemain, galeri penampilan, nama, tim, nomor, keputusan tinjauan, koreksi ID, pose, dan batas lapangan disimpan otomatis dalam `outputs/sessions/*.json.gz`. Membuka video yang sama memuat hasil itu kembali tanpa inferensi ulang. Mengganti isi file video membatalkan cache lama. Hasil pipeline lama meminta analisis ulang agar filter lapangan ketat diterapkan. Analisis ulang dapat menghasilkan fragmen berbeda dan memerlukan verifikasi lagi; roster lama tetap tersedia sebagai tujuan koreksi. Identitas berlaku **per video**, bukan antarvideo berbeda.

## Pipeline dan batas akurasi

- YOLO11s mendeteksi orang dan bola. Area lantai biru dibuat dari poligon komponen utama sehingga logo, garis, atau lubang warna di bawah pemain tidak memutus track. **Jika lapangan tidak tervalidasi, tidak ada orang yang dipublikasikan sebagai pemain**; mask kamera sebelumnya tidak dipakai. Orang bertubuh terpotong tanpa posisi kaki yang valid ditolak. Estimasi kontak sepatu dengan lantai menentukan posisi terhadap batas; pergelangan kaki saja tidak cukup. Batas otomatis tetap perkiraan dan perlu dikoreksi bila meleset dari garis sebenarnya.
- BoT-SORT dengan OSNet mempertahankan track jangka pendek. Pergantian shot mereset tracker gerak, sedangkan galeri penampilan tetap disimpan. Crop yang saling menutupi berat tidak ditambahkan ke galeri. Deteksi terlalu singkat ditandai untuk pemeriksaan dan tidak masuk tampilan aktif sampai diverifikasi.
- Pencocokan akhir memakai bukti beberapa frame, warna tim, konsistensi waktu, dan gerak kamera. Penggabungan otomatis dibatasi pada celah pendek dengan kandidat jelas. Kemiripan setelah hilang lama atau berganti kamera menghasilkan **saran tinjauan**, tidak langsung mewarisi nama. Ini dapat menghasilkan lebih banyak ID sementara dibanding pendekatan yang memaksa setiap crop ke pemain lama. Jumlah ID sementara bukan jumlah orang yang sudah pasti berbeda.
- **Belum ada jaminan identitas otomatis benar sampai akhir pertandingan.** Seragam sama, pemain tertutup, gambar kecil/buram, pergantian kamera, dan pergantian pemain tetap ambigu. Konfirmasi nomor dan koreksi fragmen adalah bagian dari alur presisi. Pembacaan nomor punggung saat ini manual; OCR nomor khusus olahraga dan ReID yang dilatih pada pertandingan ini belum disertakan.
- YOLO11s-pose memprediksi 17 titik tubuh. Pemain yang tidak mendapat pose pada gambar penuh diperiksa lagi melalui crop. Pose dipasangkan satu-ke-satu ke box pemain; tidak ada skeleton contoh yang ditempelkan.
- Dua kelompok utama jersey dihitung dengan inisialisasi yang tahan terhadap warna wasit/kiper berbeda. Kecerahan diberi bobot lebih rendah, angka putih pada jersey ditekan, dan warna di luar dua kelompok dibiarkan abu-abu. Tim tetap dapat dikoreksi manual.
- **Papan otomatis adalah perkiraan area lantai yang terlihat**, bukan koordinat lapangan penuh atau pengukuran meter. Kalibrasi empat sudut memperbaiki perspektif; gerak kamera memakai optical flow dan transformasi affine terbatas agar tidak mudah hanyut secara projektif. Pergantian shot atau estimasi gagal membatalkan penerapan kalibrasi lama. Drift masih mungkin terjadi dan perlu kalibrasi ulang. Pemain di luar frame tidak dibuatkan posisi baru.
- Bola menggunakan pembatas perpindahan dan interpolasi celah pendek; deteksi bola masih dapat meleset pada blur atau objek kecil lain. Pemain di luar frame tetap ada di daftar tetapi tidak dibuatkan posisi baru di papan.

## Model dan sumber

- `yolo11s.pt`: deteksi person + sports ball.
- `models/yolo11s-pose.pt`: model pose resmi, diunduh dari [Ultralytics assets](https://github.com/ultralytics/assets/releases/tag/v8.3.0).
- `models/osnet_x0_25_msmt17.pt`: model ReID lokal.

Dokumentasi API: [YOLO11](https://docs.ultralytics.com/models/yolo11/), [pose/keypoints](https://docs.ultralytics.com/tasks/pose/), [BoT-SORT](https://mikel-brostrom.github.io/boxmot/trackers/botsort/).

Rujukan untuk tahap presisi berikutnya: [SoccerNet GSR](https://github.com/SoccerNet/sn-gamestate) menggabungkan kalibrasi lapangan, deteksi/ReID, pengenalan nomor, dan afiliasi tim; [SportsSUSHI](https://arxiv.org/abs/2502.21242) menggabungkan tracklet dengan fitur khusus olahraga. Mengganti tracker saja belum menyelesaikan identitas pemain berseragam sama.

## Pengujian

```powershell
pip install pytest
python -m pytest -q
```

Tes meliputi penyaringan lapangan, logo/lubang mask, gambar tanpa lapangan, kontak sepatu, pergantian shot, penggabungan fragmen, konflik nomor, pemisahan identitas, deteksi terlalu singkat, serta UI dan penyimpanan. Kelulusan tes aplikasi tidak mengukur akurasi tagging pada pertandingan nyata.

Untuk mengukur akurasi, buat label manual independen yang memuat **semua pemain** pada setiap frame acuan:

```json
{"complete_frame_labels":true,"frames":[
  {"index":0,"players":[{"id":"IDN_5","box":[100,200,150,350]}]}
]}
```

Koordinat di atas hanya contoh format. Gunakan box dan identitas hasil pemeriksaan video, bukan salinan hasil model.

```powershell
python evaluate_tracking.py labels_manual.json --video "1031(2).mp4"
```

Alat ini menghitung precision/recall deteksi, identity F1 dengan pencocokan global, dan pergantian ID **pada frame yang diberi label** (IoU ≥ 0,5). Hasil disimpan ke `outputs/reference/identity-metrics.json`. Sampel jarang tidak bisa membuktikan tidak ada kesalahan di antaranya; belum ada angka akurasi pertandingan penuh tanpa ground truth tersebut.
