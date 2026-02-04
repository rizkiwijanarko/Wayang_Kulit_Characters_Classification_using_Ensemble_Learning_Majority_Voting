import os
import base64  # Pastikan ini di-import
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from scipy.stats import mode

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    model_1 = tf.keras.models.load_model('models/model_1.h5')
    model_2 = tf.keras.models.load_model('models/model_2.h5')
    model_3 = tf.keras.models.load_model('models/model_3.h5')
    MODELS = [model_1, model_2, model_3]
    print("Semua model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    MODELS = []

CLASS_LABELS = [
    'abimanyu', 'anoman', 'arjuna', 'bagong', 'baladewa', 'bima', 'buta',
    'cakil', 'durna', 'dursasana', 'duryudana', 'gareng', 'gatotkaca',
    'karna', 'kresna', 'nakula_sadewa', 'patih_sabrang', 'petruk',
    'puntadewa', 'semar', 'sengkuni', 'togog'
]


def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_ensemble(image_path):
    if not MODELS:
        return "Error: Model tidak dapat dimuat."
    processed_image = preprocess_image(image_path)
    all_predictions = [np.argmax(model.predict(processed_image), axis=1)[0] for model in MODELS]
    result = mode(all_predictions)
    final_prediction_index = result.mode
    predicted_class = CLASS_LABELS[final_prediction_index]
    return predicted_class.replace('_', ' ').title()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', error="Tidak ada file yang dipilih.")

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Gunakan blok try...finally untuk memastikan file selalu dihapus
        try:
            # 1. Simpan file sementara
            file.save(filepath)

            # 2. Lakukan prediksi
            prediction = predict_ensemble(filepath)

            # 3. Baca file gambar sebagai data biner UNTUK DIKIRIM KE HTML
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            # 4. Buat string Data URI lengkap
            image_data_url = f"data:image/jpeg;base64,{encoded_string}"

            # 5. Kirim hasil ke template. Gambar sudah ada di dalam 'image_data_url'
            return render_template('index.html', prediction=prediction, image_data_url=image_data_url)

        finally:
            # 6. Hapus file fisik dari server SETELAH semua proses selesai
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"File '{filepath}' telah dihapus.")

    return render_template('index.html')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)