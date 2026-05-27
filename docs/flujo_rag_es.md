# Flujo RAG documentado en español

Autores: Fredy Sarmiento, Manuel Caro y Jorge Bravo  
Asignatura: Aplicaciones de Aprendizaje Automatico de Maquina  
Universidad: Universidad del Rosario

## Vision como estudiantes

Este proyecto nos ayuda a entender RAG como una cadena de decisiones. No se trata solo de preguntarle a un modelo grande, sino de darle evidencia recuperada desde una fuente documental. Como alumnos aprendiendo RAG, revisamos cada etapa para saber donde puede fallar una respuesta: en la limpieza del texto, en los chunks, en los embeddings, en la recuperacion, en el prompt o en la evaluacion.

## Flujo principal

1. Preparacion del entorno: se cargan librerias de LangChain, LangGraph, OpenAI, Gemini, FAISS y RAGAS.
2. Credenciales: `OPENAI_API_KEY` y `GOOGLE_API_KEY` se leen desde variables de entorno para no guardar secretos en archivos.
3. Fuente documental: el PDF se toma como base de conocimiento.
4. Preprocesamiento: el texto se limpia para reducir ruido antes de calcular embeddings.
5. Division por capitulos: se crean documentos mas manejables para resumen y recuperacion.
6. Extraccion de citas: se buscan citas; si el PDF no conserva comillas, se usan pasajes largos como respaldo.
7. Resumenes: cada capitulo se resume para crear una vista compacta del libro.
8. Embeddings: los textos se transforman en vectores numericos.
9. FAISS: los vectores se guardan en indices locales para consultas rapidas.
10. Recuperacion: dada una pregunta, se recuperan chunks, resumenes y citas semanticamente cercanos.
11. Destilacion: un LLM filtra el contexto para quedarse con la evidencia mas relevante.
12. Generacion: el modelo responde usando el contexto, no solamente conocimiento interno.
13. Verificacion: se valida si la respuesta esta soportada por el contexto y si responde la pregunta.
14. Agente avanzado: LangGraph permite planear, escoger herramientas, recuperar, responder y replantear.
15. Evaluacion: RAGAS mide relevancia, fidelidad, similitud y recuperacion del contexto.

## Lecciones aprendidas

Un sistema RAG mejora cuando sus documentos estan bien preparados. Si los chunks son malos, la busqueda recupera evidencia pobre. Si la evidencia es pobre, el modelo puede responder con inseguridad o alucinar. Por eso el flujo incluye limpieza, indices separados, verificacion y evaluacion.

El agente avanzado muestra que RAG puede ser iterativo. Primero planea, luego recupera informacion, despues decide si puede responder o si necesita replantear. Esta parte es mas compleja, pero enseña una idea importante: una aplicacion real de RAG puede combinar busqueda, razonamiento, validacion y evaluacion.

## Prueba unitaria agregada

Se agrego `tests/test_helper_functions.py` para validar dos comportamientos:

1. El extractor de citas no devuelve una lista vacia cuando el PDF no conserva comillas.
2. La limpieza de tabulaciones funciona antes de crear embeddings.

Estas pruebas no dependen de APIs externas, por lo que pueden ejecutarse rapido con:

```powershell
.\.tools\Library\bin\micromamba.exe run -p .\.conda\complex_rag python -m pytest -q
```
