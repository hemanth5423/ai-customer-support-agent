import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticRetriever:

    def __init__(self, data_path, top_k=3):

        self.data_path = data_path
        self.top_k = top_k

        print("Loading historical conversations...")

        self.df = pd.read_csv(data_path)

        self.df = self.df.dropna(
            subset=["customer_message", "support_response"]
        ).reset_index(drop=True)

        print(f"Loaded conversations: {len(self.df)}")

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Creating embeddings...")

        self.embeddings = self.model.encode(
            self.df["customer_message"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        )

        print("Retriever ready!")

    def search(self, customer_message):

        query_embedding = self.model.encode(
            [customer_message],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        top_indices = similarities.argsort()[-self.top_k:][::-1]

        results = self.df.iloc[top_indices].copy()

        results["similarity"] = similarities[top_indices]

        return results