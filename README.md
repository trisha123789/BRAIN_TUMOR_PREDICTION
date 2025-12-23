<h1> <b>🧠 Brain Tumor Prediction Using Deep Learning</b></h1>
##📌 Project Overview

Brain tumors are one of the most critical medical conditions, and early detection plays a vital role in improving patient survival rates.
This project focuses on automatic brain tumor prediction from MRI images using Deep Learning (CNN) techniques.

The system analyzes MRI scans and classifies whether a tumor is present (and optionally its type), helping doctors and radiologists in faster and more accurate diagnosis.
<img width="332" height="878" alt="output_screenshot" src="https://github.com/user-attachments/assets/6b4ec553-524d-4842-88b6-dc4db2e95eba" />

##🎯 Objectives

Detect brain tumors from MRI images

Reduce manual effort in medical diagnosis

Improve prediction accuracy using deep learning

Build a real-world healthcare-based AI application

##🧠 Technology Stack

Programming Language: Python

Deep Learning Framework: TensorFlow / Keras

Image Processing: OpenCV

Numerical Computation: NumPy

Data Handling: Pandas

Visualization: Matplotlib

IDE: VS Code

📂 Project Structure
BRAIN_TUMOR_PREDICTION/
│
├── dataset/
│   ├── training/
│   └── testing/
│
├── model/
│   └── brain_tumor_model.h5
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore

🗂 Dataset Description

MRI brain scan images

Images are categorized into:

Tumor

No Tumor
(or multiple tumor classes depending on dataset)

The dataset is preprocessed using:

Image resizing

Normalization

Data augmentation (rotation, zoom, flip)

🧪 Model Architecture

Convolutional Neural Network (CNN)

Layers used:

Convolutional layers

MaxPooling layers

Dropout (to prevent overfitting)

Fully connected (Dense) layers

Activation Functions:

ReLU

Sigmoid / Softmax

⚙️ How It Works

Load MRI image

Preprocess image (resize, normalize)

Pass image through trained CNN model

Model predicts tumor presence

Display prediction result

▶️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/trisha123789/BRAIN_TUMOR_PREDICTION.git

2️⃣ Navigate to project directory
cd BRAIN_TUMOR_PREDICTION

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
python app.py

📊 Results

High accuracy on validation dataset

Efficient prediction for unseen MRI images

Capable of real-time or near real-time inference

(Add screenshots or accuracy values if available)

🚀 Future Enhancements

Multi-class tumor classification

Integration with web app (Streamlit / Flask)

Model optimization using Transfer Learning (MobileNet, ResNet)

Deployment on cloud

Explainable AI (Grad-CAM visualization)

🏥 Applications

Medical diagnosis assistance

Hospital decision-support systems

Healthcare AI research

Academic and major projects

👩‍💻 Author

Trisha
Engineering Student | AI & Deep Learning Enthusiast
Passionate about applying AI to real-world healthcare problems

📜 License

This project is for educational and research purposes only.
