"""Dataset loading service for Hugging Face datasets."""

from datasets import load_dataset
from app.core.config import settings


class DatasetLoaderService:
    """Service for loading and processing datasets from Hugging Face."""

    @staticmethod
    def load_tech_qa_dataset(split: str = "train"):
        """Load the tech-qa dataset from Hugging Face.

        Args:
            split: Dataset split to load (train, validation, test)

        Returns:
            Loaded dataset
        """
        print(f"Loading {settings.huggingface_dataset} dataset ({split} split)...")
        dataset = load_dataset(settings.huggingface_dataset, split=split)
        print(f"Loaded {len(dataset)} samples")
        return dataset

    @staticmethod
    def prepare_documents(dataset, question_col: str = "question", answer_col: str = "answer"):
        """Prepare documents from dataset for vectorization.

        Args:
            dataset: Loaded dataset
            question_col: Column name for questions
            answer_col: Column name for answers

        Returns:
            Tuple of (documents, metadatas, ids)
        """
        documents = []
        metadatas = []
        ids = []

        for idx, sample in enumerate(dataset):
            # Combine question and answer as document
            question = sample.get(question_col, "")
            answer = sample.get(answer_col, "")

            # Create document combining both question and answer
            doc = f"Question: {question}\nAnswer: {answer}"
            documents.append(doc)

            # Store metadata
            metadatas.append({
                "question": question,
                "answer": answer,
                "source": "tech_qa_dataset"
            })

            ids.append(f"tech_qa_{idx}")

        return documents, metadatas, ids

    @staticmethod
    def batch_documents(documents: list, batch_size: int = 100):
        """Batch documents for processing.

        Args:
            documents: List of documents
            batch_size: Size of each batch

        Yields:
            Batches of documents
        """
        for i in range(0, len(documents), batch_size):
            yield documents[i : i + batch_size]
