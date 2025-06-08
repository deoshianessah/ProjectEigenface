# Project Eigenface
Project Based Learning 1 : Aplikasi Nilai Eigen dan Eigen Face pada Pengenalan Wajah

## Anggota Kelompok 
1. Besty Mega Fauziah (L0124007)
2. Deoshi Anessah Zheren Areja (L0124009)
3. Dina Hamala Nur Rosyidah (L0124010)

## Deskripsi Program
Pengenalan wajah (Face Recognition) adalah teknologi biometrik yang bisa dipakai untuk mengidentifikasi wajah seseorang untuk berbagai kepentingan khususnya keamanan. Program pengenalan wajah melibatkan kumpulan citra wajah yang sudah disimpan pada database lalu berdasarkan kumpulan citra wajah tersebut, program dapat mempelajari bentuk wajah lalu mencocokkan antara kumpulan citra wajah yang sudah dipelajari dengan citra yang akan diidentifikasi.

## Fitur
- **Pemrosesan Dataset Otomatis** : Deteksi folder dataset berisi gambar wajah terstruktur.
- **PCA Manual** : Implementasi dari awal untuk komputasi eigenvalue dan eigenvector (tanpa sklearn).
- **Pengenalan Wajah** : Menampilkan hasil prediksi terhadap wajah uji dan wajah dataset yang paling mirip.
- **Visualisasi Eigenfaces** : Tampilkan komponen utama (Eigenfaces) hasil dari PCA.

## Teknologi yang Digunakan
- Python
- NumPy
- OpenCV
- Streamlit
- Pillow (PIL)
  
## Cara Penggunaan
1. Buka folder app.py kemudian klik open in terminal 
2. Lalu berikan command prompt streamlit run app.py / python -m streamlit run app.py
3. Tunggu beberapa detik hingga antarmuka aplikasi muncul di browser
4. Pilih folder dataset yang ingin digunakan
5. Pilih gambar uji (test image)
6. Aplikasi akan secara otomatis melakukan proses pengenalan wajah
7. Tunggu hingga hasil prediksi dan visualisasi wajah muncul di halaman utama
