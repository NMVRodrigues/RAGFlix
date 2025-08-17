import pandas as pd
import numpy as np

from langchain_core.documents import Document
from tqdm import tqdm


def create_documents_from_dataset(dataset: pd.DataFrame) -> list[Document]:
    """
    Create a list of Document objects from a dataset.

    Args:
        dataset (list): A list of dictionaries where each dictionary represents a document.

    Returns:
        list: A list of Document objects.
    """

    documents = []
    for index, item in tqdm(dataset.drop(['tmdb_id', 'imdb_id'], axis=1).iterrows(), total=dataset.shape[0], desc="Creating documents"):
        if item['overview'] == np.nan:
            continue
        content = f"Title: {item['title']}\n Release: {item['release_date']}\n Director: {item['director']}\n \
            Runtime: {item['runtime']}\n Genres: {item['genres']}\n Overview: {item['overview']}"
        metadata = item.to_dict()
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    return documents
