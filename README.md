# **Proyek Rice Classification in Turkey**
 
Course Machine Learning Terapan

Mikael Rizki Pandu Ekanto - Universitas Kristen Duta Wacana
```
while (!(succeed = try()));
```
**Find Me At :**
[Instagram](https://www.instagram.com/ignyakiki/) or
[GitHub](https://github.com/mikaelrizki) or
[LinkedIn](www.linkedin.com/in/mikaelrizki)

Proyek Predictive Analytics berikut disusun guna untuk memenuhi submission pada course Machine Learning Terapan Dicoding. Dalam proyek ini, akan dibangun sebuah model Machine Learning yang dapat mengklasifikasikan varian beras yang terdapat di Turki.

## **Domain Proyek**

### **Latar Belakang**


**Beras** merupakan salah satu produk biji-bijian yang paling banyak diproduksi di seluruh dunia. Beras juga menjadi sumber makanan pokok yang memiliki **banyak varietas genetik**. Varietas ini dipisahkan satu sama lain berdasarkan karakteristik dari masing-masing varian. Karakteristik tersebut meliputi **tekstur**, **bentuk**, dan **warna**. Dengan ciri-ciri yang membedakan varietas beras, maka dimungkinkan untuk **mengklasifikasikan** dan **mengevaluasi kualitas beras**. 

Dalam memilih suatu beras yang hendak dikonsumsi, maka kita dapat mengidentifikasi beras mana yang memiliki kualitas lebih baik dan tidak. **Kualitas** tersebut dapat terlihat dari **karakteristik biji beras** yang akan dibeli. Bahkan tidak jarang terdapat beras palsu yang juga beredar dipasaran. Oleh karena hal tersebut, diperlukan suatu **sistem** yang dapat **mengidentifikasi** dan **memprediksi varian dari suatu beras**.

<br>
<div><img src="https://cdn1-production-images-kly.akamaized.net/Cq0JGcRlrWieYHAMv_GwG5gufEM=/1200x675/smart/filters:quality(75):strip_icc():format(jpeg)/kly-media-production/medias/2914926/original/005311900_1568795429-2019-09-18.jpg" width="1000"/></div>

[Referensi gambar](https://cdn1-production-images-kly.akamaized.net/Cq0JGcRlrWieYHAMv_GwG5gufEM=/1200x675/smart/filters:quality(75):strip_icc():format(jpeg)/kly-media-production/medias/2914926/original/005311900_1568795429-2019-09-18.jpg)
<br>

Terdapat salah satu penelitian yang dapat mengklasifikasikan lima varietas padi berbeda yang sering ditanam di **Turki**, yaitu **Arborio**, **Basmati**, **Ipsala**, **Jasmine** dan **Karacadag**. Dalam penelitian tersebut terdapat sampel sebanyak **75.000 gambar beras** dengan **15.000 sampel** dari **masing-masing varietas** yang ada. Dari peneilitian tersebut didapatkan karakteristik dari sebuah beras di Turki yang memiliki 106 fitur termasuk 12 morfologi, 4 bentuk dan 90 fitur warna yang diperoleh dari gambar-gambar sample yang digunakan.

Dalam mengembangkan proses klasifikasi tersebut, maka akan dilakukan **pembangunan model machine learning untuk mengklasifikasikan varian beras yang terdapat di Turki**. Diharapkan model ini mampu memprediksi jenis beras berdasarkan gambar yang diberikan. Prediksi ini nantinya dijadikan acuan apakah beras tersebut merupakan beras yang beredar di Turki atau bukan (Beras Palsu).

Berikut adalah referensi dari **dataset** yang akan digunakan beserta beberapa **literatur penelitian klasifikasi beras di Turki**.

1. DATASET: https://www.muratkoklu.com/datasets/

2. KOKLU, M., CINAR, I. and TASPINAR, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187, 106285. DOI: https://doi.org/10.1016/j.compag.2021.106285

3. CINAR, I. and KOKLU, M. (2021). Determination of Effective and Specific Physical Features of Rice Varieties by Computer Vision In Exterior Quality Inspection. Selcuk Journal of Agriculture and Food Sciences, 35(3), 229-243. DOI: https://doi.org/10.15316/SJAFS.2021.252

4. CINAR, I. and KOKLU, M. (2022). Identification of Rice Varieties Using Machine Learning Algorithms. Journal of Agricultural Sciences, 28 (2), 307-325. DOI: https://doi.org/10.15832/ankutbd.862482

5. CINAR, I. and KOKLU, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. International Journal of Intelligent Systems and Applications in Engineering, 7(3), 188-194. DOI: https://doi.org/10.18201/ijisae.2019355381

## **Business Understanding**

Proyek ini dibangun untuk kalangan dengan karakteristik bisnis sebagai berikut :

+ Konsumen Turki yang ingin mengetahui jenis atau varian beras yang mereka miliki.
+ Konsumen yang ingin mengetahui apakah beras tersebut merupakan salah satu dari varian beras yang terdapat di Turki (Mengidentifikasi beras palsu atau bukan).

### Problem Statement

1. Fitur yang terdapat pada identifikasi biji beras?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
3. Bagaimana cara mengklasifikasikan varian beras yang terdapat di Turki?
4. Bagaimana cara mengetahui beras tersebut merupakan varian beras yang tersedia di Turki?

### Goals

1. Mengetahui fitur untuk mengidentifikasi varian suatu beras.
2. Melakukan persiapan data untuk dapat dilatih oleh model.
3. Melakukan pembangunan model berbasis Machine Learning untuk dapat mengklasifikasikan varian beras yang terdapat di Turki.
4. Membuat pengujian varian beras dalam bentuk gambar pada model Machine Learning yang telah dibangun.

### Solution Statement

1. Melakukan penelitian atau analisis data pada dataset dan literatur. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi antar fitur dan mendeteksi outlier.
2. Menyiapkan data agar bisa digunakan dalam membangun model.
3. Melakukan analisis dari model yang akan dibangun dan mencari algoritma yang paling efisien dalam mengklasifikasikan varian beras yang terdapat di Turki. 
3. Melakukan pengujian varian beras dalam bentuk gambar pada model Machine Learning yang telah dibangun dengan matriks evaluasi berdasarkan akurasi. Akurasi sangat diperlukan untuk mengetahui ketepatan pada pengujian tersebut. Pengujian tersebut juga sebagai bahan evaluasi algoritma manakah yang memiliki akurasi lebih tinggi dengan loss model yang lebih rendah. Akan dibangun 2 jenis model dengan VGG16 sebagai solusi improvement dari basis model.

## **Data Understanding**

Dataset yang digunakan dalam proyek ini merupakan data Gambar Beras *(Rice Image Dataset)* dengan berbagai lima buah klasifikasi yang berbeda. Dataset ini dapat diunduh di [Kaggle : Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset).

Berikut informasi pada dataset :

+ Dataset memiliki format gambar (.jpg).
+ Dataset memiliki 75.000 sample data gambar.
+ Dataset memiliki 5 buah kelas.
+ Tidak terdapat data gambar yang *corrupt/error*.

### Kelas pada dataset
Terdapat lima buah kelas pada dataset, yaitu sebagai berikut :
+ Varian Beras Arborio
+ Varian Beras Basmati
+ Varian Beras Ipsala
+ Varian Beras Jasmine
+ Varian Beras Karacadag

### Visualisasi Data
Berikut adalah beberapa bentuk visualisasi data yang diperoleh dari dataset.
<br>
<div><img src="Img/VisDat1.png"/><img src="Img/VisDat2.png"/><img src="Img/VisDat3.png"/></div>
<br>
Tidak hanya itu, untuk memvisualisasikan jumlah data pada datset dapat dilihat pada hasil visualisasi berikut.
<br><br>
<div><img src="Img/Dataset.png" width="400"/></div>

## **Data Preparation**

+ Import Dataset From Kaggle
  
  Langkah pertama adalah **pengunduhan dataset** sebagai bahan atau data yang akan digunakan dalam proyek Machine Learning ini. Dataset yang akan digunakan adalah dataset **Rice Image Dataset** yang dapat diperoleh melalui [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset). 
  Dataset tersebut kemudian di-*import* menggunakan **API Kaggle** dengan **API Key** yang diunggah ke dalam *Notebook*.

+ ZIP Extraction
  
  Setelah pengunduhan dataset, akan dilanjutkan dengan **ekstraksi file .zip** dari dataset tersebut karena dataset yang telah diunduh sebelumnya masih dalam bentuk ekstensi file .zip sehingga diperlukan ekstraksi pada file tersebut agar file di dalamnya dapat diakses dengan mudah.

+ Split Dataset

  Proses selanjutnya adalah **pembagian Dataset**. Dataset akan dibagi menjadi train, validation, dan test set. Data train atau latih akan digunakan untuk membangun model, sedangkan data validation dan test akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 75.000 data gambar akan dibagi menjadi **70% (Train Set Data)** dan **30% (Validation dan Test Set Data)**.

+ Labeling Dataset

  Selanjutnya diperlukan deklarasi dari kelas yang tersedia dalam dataset. Pada dataset tersebut terdapat **5 kelas beras** yaitu **Arborio, Basmati, Ipsala, Jasmine, dan Karacadag**. Setelah pendeklarasian kelas, maka akan dilanjutkan dengan **labeling dataset** menggunakan deklarasi yang telah dibuat. Labeling dataset penting untuk dilakukan agar sistem model dapat dibangun untuk mengklasifikasikan varian beras yang sudah dideklarasikan.

## **Modeling**
Berikut adalah proses pembangunan model Machine Learning pada skenario klasifikasi varian beras di Turki.
+ Augmentasi Gambar

  Augmentasi gambar merupakan sebuah teknik yang dapat digunakan untuk memperbanyak data train dengan cara menduplikasi gambar yang telah ada dengan menambahkan variasi tertentu. Berikut adalah proses augmentasi gambar pada setiap sampel di dalam dataset dengan menggunakan ImageDataGenerator.

+ Flow Train Set Data and Validation Set Data

  Selanjutnya, data train dan validasi dari kumpulan data gambar akan di-load ke dalam memori melalui fungsi flow(). Pada fungsi ini juga dapat dilakukan pendefinisian batasan pada target_size, batch_size, dan pemilihan mode yaitu 'categorical' karena model akan dibangun dengan penyelesaian Klasifikasi.

+ Pembuatan Conv2D Maxpooling Layer

  Pada proyek ini akan digunakan Conv2D Maxpooling LayerConv2D Maxpooling Layer untuk melakukan permodelan pada proyek Machine Learning dengan basis citra gambar digital. Pembangunan layer ini menggunakan mode sequential.

+ Pembangunan Model
  
  Selanjutnya akan dibangun 2 jenis model yang berbeda. Pada model awal atau bisa disebut dengan *base model* akan digunakan algoritma Machine Learning 
  pada pemrosesan gambar. Namun, pada model berikutnya akan dibagun menggunakan pengaplikasian VGG16 yang merupakan sebuah module Tensorflow untuk mempermudah suatu pekerjaan dalam pemrograman. Menggunakan VGG16 menjadi proses *improvement* terhadap model yang dibuat.

## **Evaluation**
Matriks evaluasi yang digunakan pada proyek ini adalah akurasi *(Accuracy)*. Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan varian yang sebenarnya *Validation* dan *Test Set Data*.

Dari kedua macam model yang telah dibuat, maka dapat kita bandingkan kedua hasil evaluasinya. Berikut adalah perbandingan hasil evaluasi kedua model tersebut.

<br>
<div><img src="Img/Perbandingan.png" width="400"/></div>

Dari perbandingan akurasi kedua model di atas maka dapat **disimpulkan** bahwa model dengan aplikasi **VGG16** memiliki **akurasi yang lebih baik** dengan **loss model yang lebih rendah**. Sehingga dapat disimpulkan bahwa model tersebut sebagai **solusi model yang terbaik dalam studi kasus ini**.

## **Testing Model**
Setelah berhasil membuat model, kita dapat menguji model yang telah disusun untuk memprediksi jenis atau varian dari beras berdasarkan gambarnya. Berikut adalah hasil pengujian yang dilakukan.

<br>
<div><img src="Img/Testing.png" width="450"/></div>