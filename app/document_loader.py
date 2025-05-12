import os
import docx2txt
import PyPDF2


def load_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read(), {'source': file_path}
    except Exception as e:
        print(f"Error loading TXT file: {e}")
        return "", {}


def load_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text, {'source': file_path}
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return "", {}


def load_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        return text, {'source': file_path}
    except Exception as e:
        print(f"Error loading DOCX file: {e}")
        return "", {}


def load_documents(directory):
    supported_extensions = {'.txt': load_txt, '.pdf': load_pdf, '.docx': load_docx}
    documents = []

    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        loader = supported_extensions.get(ext)
        if loader:
            content, metadata = loader(os.path.join(directory, filename))
            if content:
                documents.append({'content': content, 'metadata': metadata})

    return documents


if __name__ == "__main__":
    docs = load_documents("../documents")
    print(f"Loaded {len(docs)} documents.")
