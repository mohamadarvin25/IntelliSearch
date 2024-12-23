# IntelliSearch

Tugas Pemrograman 4 - TBI

<!-- [Link Deployment](http://34.105.89.229:8001/) -->

<!-- Karena file model yang digunakan terlalu besar, maka kami tidak menguploadnya ke github. Untuk file model dapat diunduh di [sini](https://drive.google.com/drive/folders/1DNfkZskujfYG9P5-vlvL-xxfv_YJiJnA). Setelah diunduh, file model dapat diletakkan di folder `/search/letor/` -->
 
## Anggota Kelompok

- 2006596996 Mohamad Arvin Fadriansyah 

---
Cara menjalankan aplikasi:
1. Masuk ke dalam repositori yang sudah di-_clone_ dan jalankan perintah berikut
   untuk menyalakan _virtual environment_:

   ```shell
   python -m venv env
   ```
2. Nyalakan environment dengan perintah berikut:

   ```shell
   # Windows
   .\env\Scripts\activate
   # Linux/Unix, e.g. Ubuntu, MacOS
   source env/bin/activate
   ```
3. Install dependencies yang dibutuhkan untuk menjalankan aplikasi dengan perintah berikut:

   ```shell
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi Django menggunakan server pengembangan yang berjalan secara
   lokal:

   ```shell
   python manage.py runserver
   ```
7. Bukalah `http://localhost:8000` pada browser favoritmu untuk melihat apakah aplikasi sudah berjalan dengan benar.
