from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for doc in documents:
        content = doc['content']
        metadata = doc['metadata']
        split_texts = splitter.split_text(content)

        for i, chunk in enumerate(split_texts):
            chunks.append({
                'content': chunk,
                'metadata': {
                    **metadata,
                    'chunk_index': i
                }
            })

    return chunks


if __name__ == "__main__":
    from document_loader import load_documents

    docs = load_documents("../documents")
    chunks = split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")