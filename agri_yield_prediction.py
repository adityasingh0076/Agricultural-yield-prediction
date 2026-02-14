# 🌾 AGRICULTURAL YIELD PREDICTION USING CNN + ANN
# This project predicts agricultural yield using:
# 1. Tabular data from a CSV file
# 2. Image data from crop images
# It combines CNN (for image features) and ANN (for tabular features).

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

print("🚀 Starting Agricultural Yield Prediction Model...")

# STEP 1: Load and preprocess CSV data
df = pd.read_csv("data.csv")
print("✅ Data loaded successfully!\n", df.head())

# Encode categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
df["crop"] = le_crop.fit_transform(df["crop"])
df["soil_type"] = le_soil.fit_transform(df["soil_type"])

# Separate features and target
X_tabular = df[["rainfall", "temperature", "humidity", "soil_type"]].values
y = df["yield"].values

# Normalize numerical data
scaler = StandardScaler()
X_tabular = scaler.fit_transform(X_tabular)

# Split dataset
X_train_tab, X_test_tab, y_train, y_test = train_test_split(
    X_tabular, y, test_size=0.2, random_state=42
)

# STEP 2: Load image dataset
image_folder = "images"
img_height, img_width = 128, 128

datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_images = datagen.flow_from_directory(
    image_folder,
    target_size=(img_height, img_width),
    batch_size=16,
    subset='training'
)

val_images = datagen.flow_from_directory(
    image_folder,
    target_size=(img_height, img_width),
    batch_size=16,
    subset='validation'
)

# STEP 3: Define CNN model
image_input = Input(shape=(img_height, img_width, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
cnn_output = layers.Dense(64, activation='relu')(x)

# STEP 4: Define ANN model
tab_input = Input(shape=(X_train_tab.shape[1],))
y1 = layers.Dense(64, activation='relu')(tab_input)
y1 = layers.Dense(32, activation='relu')(y1)
ann_output = layers.Dense(16, activation='relu')(y1)

# STEP 5: Combine CNN + ANN
combined = layers.concatenate([cnn_output, ann_output])
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dense(32, activation='relu')(z)
z = layers.Dense(1, activation='linear')(z)

model = Model(inputs=[image_input, tab_input], outputs=z)

# STEP 6: Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

# STEP 7: Manual Training Loop
print("\n🧠 Training the model...")

epochs = 5  # you can increase this if needed
train_mae = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_images.reset()
    total_loss, total_mae = 0, 0
    steps = len(train_images)

    for step in range(steps):
        imgs, _ = next(train_images)
        idx = np.random.randint(0, len(X_train_tab), size=len(imgs))
        X_tab_batch = X_train_tab[idx]
        y_batch = y_train[idx]

        metrics = model.train_on_batch([imgs, X_tab_batch], y_batch)
        total_loss += metrics[0]
        total_mae += metrics[1]

    avg_mae = total_mae / steps
    train_mae.append(avg_mae)
    print(f"Epoch {epoch+1} → Loss: {total_loss/steps:.4f}, MAE: {avg_mae:.4f}")

# STEP 8: Evaluate model
print("\n📊 Evaluating the model...")
val_images.reset()

try:
    val_imgs, _ = next(val_images)
except StopIteration:
    val_images.reset()
    val_imgs, _ = next(val_images)

batch_size = len(val_imgs)
if batch_size == 0:
    raise ValueError("No validation images found. Check your 'images' folder structure.")

idx = np.random.randint(0, len(X_test_tab), size=batch_size)
X_tab_val = X_test_tab[idx]
y_val = y_test[idx]

loss, mae = model.evaluate([val_imgs, X_tab_val], y_val, verbose=0)
print(f"\n✅ Model trained successfully! Final Loss: {loss:.4f}, MAE: {mae:.4f}")

# STEP 9: Plot MAE over epochs (manual history)
plt.plot(range(1, len(train_mae) + 1), train_mae, marker='o', label='Train MAE')
plt.title('Training MAE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

# STEP 10: Save model
save_path = os.path.join(os.getcwd(), "agri_yield_model.h5")
model.save(save_path)
print(f"✅ Model saved at: {save_path}")
