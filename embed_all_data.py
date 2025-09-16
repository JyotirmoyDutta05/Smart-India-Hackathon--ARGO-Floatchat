import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load the ARGO dataset
df = pd.read_csv("argo_data.csv")

# Rename important columns to match your schema
df = df.rename(columns={
    "PLATFORM_NUMBER": "float_id",
    "PRES": "pressure",
    "temperature": "temperature",
    "salinity": "salinity",
    "latitude": "latitude",
    "longitude": "longitude",
    "date": "date",
    "region": "region"
})

# Convert pressure → depth (approximate: 1 dbar ≈ 1 m)
df["depth"] = df["pressure"]

# Add placeholder columns for missing attributes
df["oxygen"] = None
df["pH"] = None
df["conductivity"] = None

# Features for numeric embeddings
features = ["latitude", "longitude", "depth", "temperature", "salinity"]

# Normalize numeric data
scaler = StandardScaler()
numeric_data = scaler.fit_transform(df[features])

# Load text embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a text description for each float
df["description"] = df.apply(
    lambda row: f"Float {row['float_id']} in {row['region']} at depth {row['depth']}m "
                f"with temperature {row['temperature']}°C and salinity {row['salinity']}.",
    axis=1
)

# Generate text embeddings
text_embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)

# Combine numeric + text embeddings
combined_embeddings = np.hstack([numeric_data, text_embeddings])

# Build FAISS index
d = combined_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(combined_embeddings.astype(np.float32))

# Save FAISS index and metadata
faiss.write_index(index, "argo_index.faiss")
with open("argo_metadata.pkl", "wb") as f:
    pickle.dump({"df": df, "scaler": scaler}, f)

print("✅ Embeddings generated and stored successfully!")
print(f"Total floats indexed: {len(df)}")
