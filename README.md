# 🌿 Plant Disease Classifier

A deep learning web application that detects **plant diseases** from leaf images using a **Convolutional Neural Network (CNN)** trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). Built with TensorFlow and Streamlit.

---

## 🚀 Demo

> Upload a photo of a plant leaf and get an instant AI diagnosis — healthy or diseased — with confidence score.

---

## 🗂️ Project Structure
## 📥 Download Trained Model
The model file is too large for GitHub. Download it here:

👉 [Download plant_disease_prediction.h5](https://drive.google.com/file/d/1BELKvWfR8-WqumPknQq1CZPdITHJNiIk/view?usp=drive_link)

After downloading, place the file in the root project folder:

```
plant-disease-classifier/
│
├── main.py                              # Streamlit web application
├── plant_disease_prediction.ipynb       # Model training notebook
├── plant_disease_prediction_model.h5    # model too large to upload here
├── class_indices.json                   # Class label mappings (38 classes)
├── requirements.txt                     # Python dependencies
├── sample_images/                       # Sample leaf images for testing
└── README.md
```

---

## 🌱 Supported Plants & Diseases (38 Classes)

| Plant       | Conditions Detected                                              |
|-------------|------------------------------------------------------------------|
| Apple       | Apple Scab, Black Rot, Cedar Apple Rust, Healthy                |
| Blueberry   | Healthy                                                          |
| Cherry      | Powdery Mildew, Healthy                                          |
| Corn        | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape       | Black Rot, Esca, Leaf Blight, Healthy                            |
| Orange      | Haunglongbing (Citrus Greening)                                  |
| Peach       | Bacterial Spot, Healthy                                          |
| Pepper      | Bacterial Spot, Healthy                                          |
| Potato      | Early Blight, Late Blight, Healthy                               |
| Raspberry   | Healthy                                                          |
| Soybean     | Healthy                                                          |
| Squash      | Powdery Mildew                                                   |
| Strawberry  | Leaf Scorch, Healthy                                             |
| Tomato      | 9 conditions including Bacterial Spot, Mosaic Virus, Healthy     |

---

## 🧠 Model Architecture

The CNN model was trained on the PlantVillage dataset using color leaf images:

- **Input:** 224×224 RGB images
- **Architecture:** CNN with Conv2D, MaxPooling, and Dense layers
- **Output:** Softmax over 38 classes
- **Framework:** TensorFlow / Keras
- **Dataset:** PlantVillage (87,000+ images across 38 classes)

> See `plant_disease_prediction.ipynb` for full training details, data augmentation, and evaluation metrics.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-classifier.git
cd plant-disease-classifier
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🖼️ How to Use

1. Launch the app with `streamlit run main.py`
2. Click **"Browse files"** and upload a `.jpg`, `.jpeg`, or `.png` leaf image
3. Click **"Classify"**
4. View the predicted disease, confidence score, and health status

---

## 📦 Dependencies

| Package     | Version  | Purpose                     |
|-------------|----------|-----------------------------|
| streamlit   | 1.32.0   | Web application framework   |
| tensorflow  | 2.15.0   | Deep learning model loading |
| numpy       | 1.26.4   | Array operations            |
| Pillow      | 10.2.0   | Image preprocessing         |

---

## 📁 Dataset

**PlantVillage Dataset** by Abdullah Ali  
- 87,000+ color images of healthy and diseased plant leaves  
- 38 classes across 14 crop species  
- License: CC-BY-NC-SA-4.0  
- [Kaggle Dataset →](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**  
- GitHub: [@YOUR_USERNAME](https://github.com/rohannp02y)
