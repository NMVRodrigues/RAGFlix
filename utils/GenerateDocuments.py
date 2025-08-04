from langchain_core.documents import Document


def create_documents_from_dataset(dataset: pd.DataFrame) -> list[Document]:
    """
    Create a list of Document objects from a dataset.

    Args:
        dataset (list): A list of dictionaries where each dictionary represents a document.

    Returns:
        list: A list of Document objects.
    """
    documents = []
    for item in dataset.drop(['tmdb_id', 'imdb_id'], axis=1,inplace=True).iterrows():
        content = f"{item['title']} : {item['overview']}"
        metadata = item.to_dict()
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    return documents
