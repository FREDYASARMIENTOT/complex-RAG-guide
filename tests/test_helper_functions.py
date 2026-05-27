from langchain_core.documents import Document

from helper_functions import extract_book_quotes_as_documents, replace_t_with_space


def test_extract_book_quotes_falls_back_to_passages_when_quotes_are_missing():
    """Verifica que RAG tenga documentos aun cuando el PDF no conserve comillas."""
    documents = [
        Document(
            page_content=(
                "Este fragmento simula una pagina extraida de un PDF donde las "
                "comillas originales se perdieron, pero el texto sigue siendo "
                "suficientemente largo para recuperarlo como evidencia."
            )
        )
    ]

    quotes = extract_book_quotes_as_documents(documents, min_length=50)

    assert len(quotes) == 1
    assert "fragmento simula una pagina" in quotes[0].page_content


def test_replace_t_with_space_cleans_tabs_in_documents():
    """Comprueba una limpieza basica antes de crear embeddings."""
    documents = [Document(page_content="Harry\taprende\tmagia")]

    cleaned = replace_t_with_space(documents)

    assert cleaned[0].page_content == "Harry aprende magia"
