🌾 Agricultural Yield Prediction using CNN + ANN
📌 Project Overview

This project predicts agricultural crop yield using a hybrid deep learning model that combines:

🖼 CNN (Convolutional Neural Network) → For crop image feature extraction

📊 ANN (Artificial Neural Network) → For tabular environmental data

🔗 Feature Fusion → Combines both to predict final yield

The model integrates visual crop data and environmental factors to generate accurate yield predictions.

🎯 Objective

To develop a deep learning system that predicts agricultural yield using:

Crop images

Rainfall data

Temperature

Humidity

Soil type

🧠 Model Architecture
Hybrid CNN + ANN Model

Image Input → CNN → Image Features

Tabular Input (CSV) → ANN → Numeric Features

Feature Fusion Layer

Dense Layers

Final Output → Predicted Yield

📂 Project Structure
Project/
│
├── agri_yield_prediction.py     # Training script
├── predict_yield.py             # Prediction + visualization
├── data.csv                     # Tabular dataset
├── agri_yield_model.h5          # Saved trained model
├── predicted_yields.csv         # Output predictions
├── yield_comparison_graph.png   # Result graph
└── images/
    ├── Rice/
    ├── Wheat/
    ├── Maize/
    ├── Jute/
    └── Sugarcane/

⚙️ Technologies Used

Python 3.10

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

Matplotlib

VS Code

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib

2️⃣ Train the Model
python agri_yield_prediction.py


This will:

Load dataset

Train CNN + ANN

Save model as agri_yield_model.h5

3️⃣ Run Prediction
python predict_yield.py


This will:

Load trained model

Predict yields for crop images

Save results in predicted_yields.csv

Generate yield_comparison_graph.png

📊 Output

Mean Absolute Error (MAE) during training

Predicted yield values per crop

Bar graph comparing average predicted yields

CSV file containing predictions

📈 Sample Results
Crop	Predicted Yield
Rice	4.2
Wheat	5.0
Maize	3.8
Jute	6.1
Sugarcane	7.9

(Values vary depending on dataset)

🔮 Future Improvements

Real-time prediction using IoT sensors

Integration with weather APIs

Web-based deployment

Mobile app for farmers

Satellite/drone image integration

👨‍💻 Author
Aditya Singh
Agricultural Yield Prediction Project
Deep Learning (CNN + ANN)

📜 License

This project is for academic and educational purposes.

⭐ If you like this project, give it a star!
