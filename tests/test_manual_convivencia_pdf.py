"""
Prueba unitaria para validar que el PDF del Manual de Convivencia
se carga y procesa correctamente en el pipeline RAG.
"""
import os
import sys

# Asegurar que el directorio raiz esta en el path para importar helper_functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from helper_functions import replace_t_with_space, extract_book_quotes_as_documents

PDF_PATH = "MANUAL DE CONVIVENCIA CONJUNTO RESIDENCIAL AMARANTO CLUB HOUSE.pdf"


def test_manual_convivencia_pdf_exists():
    """Verifica que el archivo PDF del Manual de Convivencia existe en el directorio."""
    assert os.path.exists(PDF_PATH), (
        f"El archivo '{PDF_PATH}' no existe en el directorio actual. "
        f"Directorio actual: {os.getcwd()}"
    )


def test_manual_convivencia_pdf_loads_successfully():
    """Verifica que PyPDFLoader carga el PDF sin errores y produce paginas."""
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    assert len(documents) > 0, (
        f"El PDF '{PDF_PATH}' no produjo ninguna pagina. "
        "Verifica que el archivo no este corrupto."
    )

    # Verificar que cada pagina tiene contenido textual
    for i, doc in enumerate(documents):
        assert hasattr(doc, 'page_content'), (
            f"La pagina {i} no tiene el atributo 'page_content'."
        )
        assert isinstance(doc.page_content, str), (
            f"El contenido de la pagina {i} no es un string."
        )


def test_manual_convivencia_pdf_has_expected_content():
    """Verifica que el PDF contiene terminos esperados del Manual de Convivencia."""
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Concatenar todo el texto extraido
    full_text = " ".join(doc.page_content.lower() for doc in documents)

    # Terminos que deberian aparecer en un manual de convivencia
    expected_terms = [
        "convivencia",
        "conjunto",
        "residencial",
        "amaranto",
    ]

    found_terms = []
    missing_terms = []

    for term in expected_terms:
        if term in full_text:
            found_terms.append(term)
        else:
            missing_terms.append(term)

    assert len(found_terms) >= 2, (
        f"Solo se encontraron {len(found_terms)} de {len(expected_terms)} terminos esperados. "
        f"Encontrados: {found_terms}. Faltantes: {missing_terms}. "
        f"El PDF puede no ser el esperado o tener problemas de extraccion de texto."
    )


def test_manual_convivencia_cleaning_works():
    """Verifica que las funciones de limpieza funcionan con el nuevo PDF."""
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Aplicar limpieza de tabulaciones
    cleaned = replace_t_with_space(documents)

    assert len(cleaned) == len(documents), (
        "La funcion replace_t_with_space altero el numero de documentos."
    )

    # Verificar que no quedan tabulaciones
    for doc in cleaned:
        assert '\t' not in doc.page_content, (
            "Quedaron tabulaciones sin limpiar en el documento."
        )


def test_manual_convivencia_extract_quotes_fallback():
    """Verifica que extract_book_quotes_as_documents funciona con el nuevo PDF
    usando el mecanismo de fallback cuando no encuentra comillas."""
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    documents = replace_t_with_space(documents)

    # Extraer citas/pasajes
    quotes = extract_book_quotes_as_documents(documents, min_length=50)

    assert len(quotes) > 0, (
        "La extraccion de citas/pasajes no produjo ningun documento. "
        "El pipeline RAG necesita al menos un documento para funcionar."
    )

    # Verificar que cada cita/pasaje tiene contenido valido
    for i, quote in enumerate(quotes):
        assert len(quote.page_content.strip()) > 0, (
            f"El pasaje {i} esta vacio."
        )