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
            show_progress_bar=True
        )

        print("Semantic retriever is ready.")

    def search(self, customer_message):

        query_embedding = self.model.encode(
            [customer_message]
        )

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        top_indexes = similarities.argsort()[
            -self.top_k:
        ][::-1]

        results = self.df.iloc[top_indexes].copy()

        results["similarity"] = similarities[top_indexes]

        return results


if __name__ == "__main__":

    DATA_PATH = "data/processed/apple_support_context.csv"

    retriever = SemanticRetriever(
        DATA_PATH,
        top_k=3
    )

    customer_message = input(
        "\nEnter customer message: "
    )

    results = retriever.search(
        customer_message
    )

    print("\n" + "=" * 70)
    print("SEMANTIC RETRIEVAL RESULTS")
    print("=" * 70)

    for i, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        print(f"\n--- RESULT {i} ---")

        print("\nCUSTOMER:")
        print(row["customer_message"])

        print("\nHISTORICAL APPLE RESPONSE:")
        print(row["support_response"])

        print(
            f"\nSIMILARITY: "
            f"{row['similarity']:.3f}"
        )