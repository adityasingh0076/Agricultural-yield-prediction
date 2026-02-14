import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("agri_yield_model.h5")
print("✅ Model loaded successfully!\n")

# Base directory containing crop folders
base_dir = "images"
crops = ["Rice", "Jute", "Maize", "Sugarcane", "Wheat"]

# Dummy features if your model expects numeric input
dummy_features = np.zeros((1, 4))

# Store results for CSV + graph
results = []

# Loop through crop folders
for crop in crops:
    crop_path = os.path.join(base_dir, crop)
    if not os.path.exists(crop_path):
        print(f"⚠️ Folder not found: {crop_path}")
        continue

    print(f"🌾 Predicting yields for {crop} images...\n")
    crop_predictions = []

    for img_name in os.listdir(crop_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(crop_path, img_name)
            try:
                # Preprocess image
                img = image.load_img(img_path, target_size=(128, 128))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = model.predict([img_array, dummy_features], verbose=0)
                predicted_yield = float(prediction[0][0])
                crop_predictions.append(predicted_yield)

                # Store in results list
                results.append({"Crop": crop, "Image": img_name, "Predicted_Yield": predicted_yield})

            except Exception as e:
                print(f"❌ Error with {img_name}: {e}")

    if crop_predictions:
        avg_yield = np.mean(crop_predictions)
        print(f"✅ Average predicted yield for {crop}: {avg_yield:.2f}\n")

print("✅ All predictions completed successfully!\n")

# Save predictions to CSV
df = pd.DataFrame(results)
df.to_csv("predicted_yields.csv", index=False)
print("📁 Results saved to 'predicted_yields.csv'")

# --- Generate Graph ---
avg_yields = df.groupby("Crop")["Predicted_Yield"].mean()
plt.figure(figsize=(8, 5))
avg_yields.plot(kind="bar", color="green", edgecolor="black")
plt.title("Average Predicted Yield per Crop")
plt.xlabel("Crop")
plt.ylabel("Predicted Yield")
plt.tight_layout()
plt.savefig("yield_comparison_graph.png")
plt.show()

print("📊 Graph generated and saved as 'yield_comparison_graph.png'")
