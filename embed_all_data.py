import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss
import pickle

df = pd.read_csv("argo_dummy.csv")

features = ["latitude", "longitude", "depth", "temperature", "salinity"]

scaler = StandardScaler()
numeric_data = scaler.fit_transform(df[features])

model = SentenceTransformer("all-MiniLM-L6-v2")

df["description"] = df.apply(
    lambda row: f"Float {row['float_id']} in {row['region']} at depth {row['depth']}m "
                f"with temperature {row['temperature']}°C and salinity {row['salinity']}.",
    axis=1
)

text_embeddings = model.encode(df["description"].tolist())

combined_embeddings = np.hstack([numeric_data, text_embeddings])

d = combined_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(combined_embeddings.astype(np.float32))

faiss.write_index(index, "argo_index.faiss")
with open("argo_metadata.pkl", "wb") as f:
    pickle.dump({"df": df, "scaler": scaler}, f)

print("✅ Embeddings generated and stored successfully!")
print(f"Total floats indexed: {len(df)}")
