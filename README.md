# Proyecto RAG: Sistema de Recuperación Aumentada para Consultas sobre el Manual de Convivencia – Amaranto Club House

**Autores:** Fredy Sarmiento, Manuel Caro y Jorge Bravo  
**Asignatura:** Aplicaciones de Aprendizaje Automático de Máquina  
**Programa:** Maestría en Ingeniería de Tecnologías de la Información y Comunicación (ICT)  
**Universidad:** Universidad del Rosario  

---

## Resumen Académico para la Docente

El presente proyecto implementa un pipeline de *Retrieval-Augmented Generation* (RAG) de nivel producción, aplicado al dominio jurídico-residencial del **Manual de Convivencia del Conjunto Residencial Amaranto Club House**. La arquitectura integra:

1. **Ingesta híbrida de documentos:** fragmentación tradicional con solapamiento (*chunking*) y división lógica por capítulos del manual, complementada con extracción de normas y sanciones específicas del régimen de propiedad horizontal.
2. **Limpieza y normalización de texto:** remoción de tabuladores, colapso de saltos de línea destructivos y normalización de espacios para optimizar el conteo de tokens en los embeddings.
3. **Codificación vectorial múltiple:** se construyen tres índices FAISS independientes —fragmentos (*chunks*), resúmenes de capítulos y citas normativas— utilizando embeddings de OpenAI.
4. **Orquestación basada en agentes con LangGraph:** el sistema opera como un grafo de estado cíclico compuesto por nodos de anonimización de preguntas, planificación, desglose de tareas, selección dinámica de herramientas (*retrieve chunks*, *retrieve summaries*, *retrieve quotes*, *answer*), re-planificación y generación de respuesta final.
5. **Capa anti-alucinaciones:** verificación explícita de *groundedness* (fidelidad fáctica) mediante un fact-checker que asegura que toda respuesta esté soportada estrictamente por el contenido del Manual de Amaranto. Si no se encuentra evidencia, el sistema responde: *"Información no encontrada en el Manual de Convivencia de Amaranto Club House"*.
6. **Evaluación con RAGAS:** se miden las métricas de *Answer Correctness*, *Faithfulness*, *Answer Relevancy*, *Context Recall* y *Answer Similarity* sobre un conjunto de preguntas sintéticas formuladas desde la perspectiva de copropietarios del conjunto residencial.

El proyecto demuestra que un pipeline RAG robusto para dominios legales/residenciales requiere no solo una recuperación semántica precisa, sino también mecanismos de planificación dinámica, verificación de fidelidad y evaluación sistemática. Cada componente está documentado para que el flujo sea comprensible para estudiantes que están aprendiendo RAG.

---

## Tabla de Contenidos

- [Comprensión de Nuestro Pipeline RAG](#comprensión-de-nuestro-pipeline-rag)
- [Configuración del Entorno](#configuración-del-entorno)
- [Fragmentación de los Datos (Formas Tradicionales y Lógicas)](#fragmentación-de-los-datos-formas-tradicionales-y-lógicas)
- [Limpieza de los Datos](#limpieza-de-los-datos)
- [Reestructuración de los Datos](#reestructuración-de-los-datos)
- [Vectorización de los Datos](#vectorización-de-los-datos)
- [Creación de un Recuperador de Contexto](#creación-de-un-recuperador-de-contexto)
- [Filtro de Información Irrelevante](#filtro-de-información-irrelevante)
- [Reescritor de Consultas](#reescritor-de-consultas)
- [Razonamiento por Cadena de Pensamiento (CoT)](#razonamiento-por-cadena-de-pensamiento-cot)
- [Verificación de Relevancia y Fidelidad a los Hechos](#verificación-de-relevancia-y-fidelidad-a-los-hechos)
- [Prueba de Nuestro Pipeline RAG](#prueba-de-nuestro-pipeline-rag)
- [Visualización del Pipeline RAG con LangGraph](#visualización-del-pipeline-rag-con-langgraph)
- [Enfoque de Subgrafos y Verificación de Destilación](#enfoque-de-subgrafos-y-verificación-de-destilación)
- [Creación del Subgrafo de Recuperación y Destilación](#creación-del-subgrafo-de-recuperación-y-destilación)
- [Creación del Subgrafo para Mitigación de Alucinaciones](#creación-del-subgrafo-para-mitigación-de-alucinaciones)
- [Creación y Prueba del Ejecutor de Planes](#creación-y-prueba-del-ejecutor-de-planes)
- [Lógica de Re-Planificación](#lógica-de-re-planificación)
- [Creación del Manejador de Tareas](#creación-del-manejador-de-tareas)
- [Anonimización/Des-Anonimización de la Pregunta](#anonimizacióndes-anonimización-de-la-pregunta)
- [Compilación y Visualización del Pipeline RAG Completo](#compilación-y-visualización-del-pipeline-rag-completo)
- [Prueba del Pipeline Finalizado](#prueba-del-pipeline-finalizado)
- [Evaluación con RAGAS](#evaluación-con-ragas)
- [Resumen General](#resumen-general)

---

## Comprensión de Nuestro Pipeline RAG

Antes de comenzar a codificar, es fundamental visualizar cómo se estructura nuestro pipeline RAG. A medida que avancemos, iremos desglosando y visualizando cada uno de sus componentes.

![Pipeline RAG Completo](https://miro.medium.com/v2/resize:fit:2000/1*-neV95FnEltYAuvIE3w8cQ.png)

**Explicación del diagrama en español:** El flujo inicia con la anonimización de la pregunta del usuario (por ejemplo, un copropietario pregunta *"¿Puedo hacer un asado el domingo y qué pasa si mi perro ladra en la noche?"*). Esta pregunta se anonimiza reemplazando entidades específicas (nombres, números de apartamento) por variables para evitar sesgos del LLM. Luego, el **Planificador** construye una estrategia de alto nivel, desglosando la consulta en sub-tareas. El plan se des-anonimiza y se refina en tareas ejecutables, que el **Manejador de Tareas** asigna a las herramientas disponibles: búsqueda en fragmentos del manual (*chunks*), búsqueda en resúmenes de capítulos, búsqueda en citas normativas, o respuesta directa desde el contexto agregado. Tras cada recuperación, el **Re-Planificador** evalúa si la información acumulada es suficiente para responder o si se requiere una nueva iteración. Finalmente, cuando el sistema determina que la pregunta puede ser respondida, se genera la respuesta final y se evalúa con RAGAS.

En este proyecto, la fuente documental es el **Manual de Convivencia del Conjunto Residencial Amaranto Club House**, que contiene derechos, deberes, obligaciones económicas, uso de zonas sociales (Club House), bienes comunes, tenencia de mascotas, mudanzas, régimen de sanciones y comité de convivencia.

---

## Configuración del Entorno

LangChain, LangGraph y los demás módulos para crear un sistema RAG constituyen una arquitectura completa. Importamos los módulos únicamente cuando son necesarios, lo que facilita un aprendizaje estructurado.

El primer paso es configurar las variables de entorno que almacenarán información sensible como las claves API.

```python
# Configuración de claves para OpenAI y Google Gemini
# Leemos la clave de OpenAI desde el entorno.
# No se escribe la clave en el notebook para evitar exponer credenciales.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Leemos la clave de Google Gemini desde el entorno.
# Esta variable se entrega al modelo Gemini cuando el notebook compara modelos.
google_api_key = os.getenv('GOOGLE_API_KEY')
```

Utilizamos dos proveedores de modelos de IA:

- **OpenAI**: para embeddings (`text-embedding-ada-002`) y para los modelos de lenguaje en las cadenas de respuesta y evaluación (GPT-4o, GPT-3.5-turbo).
- **Google Gemini**: como modelo comparativo en las etapas de reescritura de preguntas, verificación de relevancia y verificación de destilación, aprovechando su capacidad de salida estructurada.

---

## Fragmentación de los Datos (Formas Tradicionales y Lógicas)

Para un sistema RAG aplicado al dominio residencial, la fuente documental es el PDF del Manual de Convivencia de Amaranto. El desafío con documentos legales/residenciales es que contienen estructura jerárquica (títulos, capítulos, artículos, parágrafos) que debe preservarse para una recuperación precisa.

Definimos la ruta del PDF:

```python
# Definimos la ruta del archivo PDF que será la fuente principal de conocimiento.
# En RAG, esta fuente documental es la evidencia que el modelo debe consultar antes de responder.
hp_pdf_path = "MANUAL DE CONVIVENCIA CONJUNTO RESIDENCIAL AMARANTO CLUB HOUSE.pdf"
```

El paso más importante antes del preprocesamiento es dividir el documento de forma **lógica** y **tradicional**:

### División Lógica por Capítulos

El Manual de Convivencia está organizado en capítulos (Derechos, Deberes, Uso de Zonas Sociales, Mascotas, Sanciones, etc.), lo cual constituye el punto de quiebre lógico ideal.

Utilizamos una función auxiliar `split_into_chapters` que detecta los encabezados de capítulo y separa el contenido en documentos individuales, cada uno con su metadata correspondiente.

### Extracción de Citas y Pasajes Normativos

El segundo punto de quiebre lógico son las citas literales del manual (normas, artículos, sanciones específicas). Si el PDF no conserva comillas tipográficas, se utilizan pasajes largos como respaldo.

### Fragmentación Tradicional (Chunking)

La fragmentación por chunks con solapamiento es el método más común. Utilizamos `RecursiveCharacterTextSplitter` con `chunk_size=1000` y `chunk_overlap=200` para que cada fragmento mantenga una relación contextual con el anterior.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)
texts = text_splitter.split_documents(documents)
```

Al finalizar esta etapa, tenemos tres formas complementarias de los datos: **capítulos**, **citas normativas** y **chunks con solapamiento**.

---

## Limpieza de los Datos

Los PDFs, especialmente los documentos legales escaneados o convertidos, contienen artefactos de formato que degradan la calidad de los embeddings. En el Manual de Amaranto encontramos:

- **Tabulaciones (`\t`)** que introducen espacios excesivos.
- **Saltos de línea múltiples** que inflan artificialmente el conteo de tokens.
- **Palabras partidas** entre líneas por la diagramación del PDF.

Realizamos una limpieza en tres pasos:

1. **Reemplazo de tabulaciones:** `tab_pattern.sub(' ', doc.page_content)`
2. **Colapso de múltiples saltos de línea:** `multiple_newlines_pattern.sub('\n', page_content)`
3. **Unión de palabras partidas:** `word_split_newline_pattern.sub(r'\1\2', page_content)`

Esta limpieza se aplica tanto a los capítulos como a los chunks tradicionales, asegurando que el texto que alimenta los embeddings sea limpio y normalizado.

---

## Reestructuración de los Datos

Los capítulos del Manual de Convivencia pueden ser extensos. Para crear una vista compacta del contenido normativo, utilizamos un LLM para generar resúmenes extensivos de cada capítulo.

Definimos un prompt template para la summarización:

```python
summarization_prompt_template = """Write an extensive summary of the following:

{text}

SUMMARY:"""
```

La cadena de summarización utiliza dos estrategias según el tamaño del capítulo:

- **`stuff`**: para capítulos cuya longitud está dentro de la ventana de contexto del modelo (GPT-3.5-turbo, ~16K tokens). Concatena todo el texto y lo resume en una sola pasada.
- **`map_reduce`**: para capítulos más largos. Resume cada sección individualmente ("map") y luego combina esos resúmenes en uno final ("reduce").

Los resúmenes se almacenan en caché (pickle) para evitar re-ejecutar el LLM en cada prueba, ahorrando tokens y tiempo de procesamiento.

---

## Vectorización de los Datos

![Vectorización](https://miro.medium.com/v2/resize:fit:2000/1*nKUaisjjiOYyInJQcg9Z7g.png)

**Explicación del diagrama en español:** El proceso de vectorización transforma cada fragmento de texto (chunks, resúmenes y citas) en un vector numérico de alta dimensionalidad utilizando el modelo de embeddings de OpenAI (`text-embedding-ada-002`). Estos vectores capturan la similitud semántica: textos con significados similares quedan cercanos en el espacio vectorial. Los vectores se almacenan en índices FAISS, que permiten búsquedas rápidas por producto interno (IndexFlatIP). Se construyen tres índices independientes: uno para chunks del manual, otro para resúmenes de capítulos, y un tercero para citas normativas. Esta separación permite que el agente seleccione dinámicamente qué fuente consultar según la naturaleza de la tarea.

Para el almacenamiento, utilizamos [FAISS](https://github.com/facebookresearch/faiss) de Meta, reconocido por su alta eficiencia en búsqueda de similitud y compatible con múltiples plataformas de bases de datos vectoriales en la nube.

Creamos tres índices vectoriales independientes:

```python
# Fragmentos del manual (chunks)
chunks_vector_store = encode_book(hp_pdf_path, chunk_size=1000, chunk_overlap=200)

# Resúmenes de capítulos
chapter_summaries_vector_store = encode_chapter_summaries(chapter_summaries)

# Citas y pasajes normativos
book_quotes_vectorstore = encode_quotes(book_quotes_list)
```

Cada índice se persiste localmente con `save_local()` para no recalcular los embeddings en cada ejecución. En futuras ejecuciones, se cargan directamente desde disco:

```python
chunks_vector_store = FAISS.load_local("chunks_vector_store", embeddings, allow_dangerous_deserialization=True)
```

---

## Creación de un Recuperador de Contexto

![Flujo del Recuperador](https://miro.medium.com/v2/resize:fit:2000/1*kgPsv7OcbtRYSDg5G_00lg.png)

**Explicación del diagrama en español:** El recuperador de contexto actúa como la capa de acceso a la base de conocimiento. Dada una pregunta de un copropietario (por ejemplo, *"¿Cuáles son las sanciones por ruido en horas de la noche?"*), el sistema consulta simultáneamente los tres índices FAISS: recupera el fragmento (chunk) más relevante del manual, el resumen del capítulo más pertinente, y las 10 citas normativas más cercanas semánticamente. Toda esta evidencia se agrega en un único contexto textual que alimentará las etapas posteriores de filtrado y generación. La función `retrieve_context_per_question` orquesta esta agregación multi-fuente, escapando caracteres problemáticos para garantizar la integridad del prompt.

Transformamos cada índice vectorial en un recuperador configurando el parámetro `k` (cantidad de documentos a retornar):

```python
# Recuperador de fragmentos del manual (top 1 más relevante)
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 1})

# Recuperador de resúmenes de capítulos (top 1 más relevante)
chapter_summaries_query_retriever = chapter_summaries_vector_store.as_retriever(search_kwargs={"k": 1})

# Recuperador de citas normativas (top 10 más relevantes, por ser textos cortos)
book_quotes_query_retriever = book_quotes_vectorstore.as_retriever(search_kwargs={"k": 10})
```

La función `retrieve_context_per_question` agrega el contexto de las tres fuentes:

1. Fragmentos del manual (información general y contexto amplio).
2. Resúmenes de capítulos (visión compacta con cita del capítulo fuente).
3. Citas normativas (evidencia textual específica de artículos y sanciones).

El contexto agregado se escapa para evitar problemas con caracteres especiales en los prompts posteriores.

---

## Filtro de Información Irrelevante

![Filtro](https://miro.medium.com/v2/resize:fit:2400/1*dUKpQkySDodwv3mdMEQxcw.png)

**Explicación del diagrama en español:** El filtro de contenido irrelevante actúa como una capa de destilación semántica. Incluso con una buena recuperación vectorial, parte del contexto recuperado puede no ser pertinente para responder la consulta específica. Este componente utiliza un LLM (GPT-4o) con una cadena de filtrado que recibe la consulta original y los documentos recuperados, y produce exclusivamente el subconjunto de información relevante. La salida está estructurada mediante un modelo Pydantic (`KeepRelevantContent`) que garantiza que solo se retorne contenido presente en los documentos originales, sin añadir información nueva. Esta etapa es crítica para evitar que información tangencial del manual (por ejemplo, normas de piscina cuando se pregunta sobre mascotas) contamine el contexto que alimenta al generador de respuestas.

Incluso después de una buena recuperación vectorial, parte del contexto puede no ser relevante. Implementamos un filtro basado en LLM que conserva únicamente la información pertinente a la consulta:

```python
keep_only_relevant_content_prompt_template = """
You receive a query: {query} and retrieved documents: {retrieved_documents} from a vector store.
You need to filter out all the non relevant information that doesn't supply important information regarding the {query}.
Your goal is just to filter out the non relevant information.
You can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
Output the filtered relevant content.
"""
```

La cadena utiliza GPT-4o con salida estructurada (`KeepRelevantContent`) para garantizar que el LLM no introduzca información nueva (**restricción crítica anti-alucinaciones**). Si el LLM no devuelve contenido estructurado válido, el sistema utiliza el contexto original como respaldo (*fallback*).

---

## Reescritor de Consultas

Uno de los desafíos en RAG es que las consultas de los usuarios no siempre son lo suficientemente descriptivas para recuperar contenido relevante. Por ejemplo, un copropietario podría preguntar: *"¿Puedo hacer una fiesta?"* cuando en realidad necesita información sobre reserva de zonas sociales, límites de horario, niveles de ruido permitidos y aforo máximo.

Para abordar esto, el **reescritor de consultas** utiliza Gemini 2.5 Pro para reformular la pregunta original, analizando la intención semántica subyacente y expandiéndola para optimizar la recuperación vectorial:

```python
rewrite_prompt_template = """
You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval.
Analyze the input question {question} and try to reason about the underlying semantic intent / meaning.
{format_instructions}
"""
```

La pregunta reescrita se convierte en la nueva entrada para el recuperador, mejorando la calidad del contexto recuperado en iteraciones subsecuentes.

---

## Razonamiento por Cadena de Pensamiento (CoT)

![Razonamiento CoT](https://miro.medium.com/v2/resize:fit:2400/1*11iiXEhOR6GqVWn_WkorYQ.png)

**Explicación del diagrama en español:** En lugar de pedir al LLM que responda directamente, el razonamiento por Cadena de Pensamiento (Chain-of-Thought, CoT) guía al modelo para que descomponga el problema en pasos intermedios antes de emitir la respuesta final. El prompt incluye ejemplos *few-shot* que ilustran el estilo de razonamiento esperado: identificar la información relevante en el contexto, encadenar inferencias lógicas y solo entonces formular la conclusión. Para el dominio de Amaranto, esto es particularmente útil en consultas que requieren cruzar múltiples secciones del manual: por ejemplo, si un residente pregunta *"¿Puedo usar el salón social para un evento familiar y cuál es el procedimiento?"*, el modelo debe razonar sobre requisitos de reserva, horarios permitidos, cobros asociados y normas de uso, integrando información de distintos capítulos del manual.

En lugar de solicitar al LLM que responda directamente, utilizamos un enfoque de razonamiento paso a paso (*Chain of Thought*). Implementamos un enfoque **few-shot CoT** donde proporcionamos al LLM múltiples ejemplos que demuestran la estructura de razonamiento deseada:

1. **Ejemplo 1:** Razonamiento comparativo (altura de personas).
2. **Ejemplo 2:** Razonamiento sobre capacidades (hechizos mágicos).
3. **Ejemplo 3:** Identificación de información faltante ("no hay suficiente contexto").

El prompt instruye al modelo a mostrar su proceso de razonamiento antes de emitir la respuesta final:

```
Context:
{context}
Question:
{question}
```

La respuesta se estructura mediante el modelo Pydantic `QuestionAnswerFromContext`, que garantiza que el output contenga la respuesta basada exclusivamente en el contenido del contexto.

---

## Verificación de Relevancia y Fidelidad a los Hechos

Una vez filtrados los documentos relevantes, realizamos una **verificación en dos etapas**:

### 1. Verificación de Relevancia

El LLM evalúa si cada documento recuperado es pertinente a la consulta. Se utiliza Gemini 2.5 Pro con salida JSON estructurada (`Relevance`):

```python
class Relevance(BaseModel):
    is_relevant: bool = Field(description="Whether the document is relevant to the query.")
    explanation: str = Field(description="An explanation of why the document is relevant or not.")
```

### 2. Verificación de Fidelidad a los Hechos (Fact-Checking)

El componente más crítico para el dominio legal/residencial: un verificador de hechos que determina si la respuesta generada está fundamentada en el contexto proporcionado por el Manual de Amaranto.

```python
class IsGroundedOnFacts(BaseModel):
    grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
```

La función `grade_generation_v_documents_and_question` combina ambas verificaciones:

- Si la respuesta **no está fundamentada** en el contexto → se etiqueta como `"hallucination"` y se reintenta la generación.
- Si está fundamentada pero **no responde completamente** la pregunta → se etiqueta como `"not_useful"` y se reescribe la pregunta para una nueva iteración.
- Si está fundamentada **y responde completamente** → se etiqueta como `"useful"` y se finaliza el flujo.

**Restricción de blindaje legal:** Si el contexto recuperado no contiene la información necesaria, el sistema responde: *"Información no encontrada en el Manual de Convivencia de Amaranto Club House"*, activando el flujo de fin de nodo (`__end__`) para evitar alucinaciones en un entorno jurídico/residencial.

---

## Prueba de Nuestro Pipeline RAG

![Pipeline RAG Central](https://miro.medium.com/v2/resize:fit:2000/1*EKCwbxNaRfBf2tGTfuRA2A.png)

**Explicación del diagrama en español:** El grafo del pipeline RAG central conecta cuatro nodos en un flujo condicional. Partiendo de la pregunta del usuario, el nodo `retrieve_context_per_question` consulta los tres índices FAISS y agrega el contexto. El nodo `keep_only_relevant_content` filtra la información no pertinente. A continuación, una arista condicional evalúa la relevancia: si el contenido es relevante, se pasa al nodo `answer_question_from_context` que genera la respuesta usando CoT; si no es relevante, se activa el nodo `rewrite_question` que reformula la consulta y retroalimenta el recuperador. Tras generar la respuesta, otra arista condicional ejecuta `grade_generation_v_documents_and_question`: si hay alucinación, se reintenta la respuesta; si no es útil, se reescribe la pregunta; si es útil, el flujo termina. Este diseño cíclico permite que el sistema refine iterativamente la respuesta hasta alcanzar un resultado fundamentado y completo.

Probamos el pipeline con una pregunta sencilla sobre el manual. El flujo completo es:

```
Pregunta → Recuperar contexto → Filtrar contenido relevante →
Verificar relevancia → Responder con CoT → Verificar fidelidad y completitud →
[Si alucinación: reintentar] [Si no útil: reescribir] [Si útil: finalizar]
```

---

## Visualización del Pipeline RAG con LangGraph

![Pipeline RAG en LangGraph](https://miro.medium.com/v2/resize:fit:1154/1*DnesQniliMnb-xlVVH9iig.png)

**Explicación del diagrama en español:** Este grafo de LangGraph visualiza el pipeline RAG como un grafo de estado cíclico. Los nodos representan funciones de procesamiento (recuperar, filtrar, reescribir, responder) y las aristas representan transiciones. Las aristas continuas indican flujo secuencial obligatorio, mientras que las aristas punteadas representan decisiones condicionales. El punto de entrada es `retrieve`, desde donde el flujo avanza a `filter`. En `filter`, una decisión condicional evalúa la relevancia: si es relevante, avanza a `answer`; si no, deriva a `rewrite`, que reformula la pregunta y retroalimenta a `retrieve` creando un ciclo de refinamiento. En `answer`, otra decisión evalúa la calidad de la respuesta: si es útil, el flujo termina; si es alucinación, se reintenta la respuesta; si no es útil, se reescribe la pregunta. Este diseño implementa un sistema de recuperación aumentada con capacidad de auto-corrección.

LangGraph nos permite visualizar el pipeline como un grafo dirigido. El grafo muestra:

- **Entrada:** `retrieve_context_per_question` (recuperación de contexto multi-fuente).
- **Procesamiento:** `keep_only_relevant_content` → verificación de relevancia → `answer_question_from_context`.
- **Ciclo de refinamiento:** Si el contenido no es relevante, se activa `rewrite_question` que realimenta la recuperación.
- **Auto-corrección:** Si la respuesta es alucinación, se reintenta; si no es útil, se reescribe la pregunta.

---

## Enfoque de Subgrafos y Verificación de Destilación

En aplicaciones reales, especialmente en el dominio legal/residencial del Manual de Amaranto, las consultas complejas requieren razonamiento multi-paso. El **enfoque de subgrafos** descompone el pipeline principal en módulos especializados:

- **Subgrafo de recuperación y destilación:** para cada fuente de datos (chunks, resúmenes, citas).
- **Subgrafo de respuesta anti-alucinaciones:** verifica que la respuesta esté fundamentada antes de emitirla.

Para la **destilación**, implementamos una verificación adicional que asegura que el contenido destilado (filtrado) esté fundamentado en el contexto original:

```python
class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(description="Whether the distilled content is grounded on the original context.")
    explanation: str = Field(description="An explanation of why the distilled content is or is not grounded.")
```

Si el contenido destilado no está fundamentado en el contexto original, se reintenta el filtrado, creando un ciclo de verificación hasta obtener contenido fiel.

---

## Creación del Subgrafo de Recuperación y Destilación

![Subgrafos de Recuperación](https://miro.medium.com/v2/resize:fit:2000/1*pHZY1YFOdJyR_8-HDswE2A.png)

**Explicación del diagrama en español:** Se construyen tres subgrafos independientes, uno para cada tipo de fuente documental: fragmentos del manual (*chunks*), resúmenes de capítulos y citas normativas. Cada subgrafo sigue el mismo patrón estructural: un nodo de recuperación específico (`retrieve_chunks_context_per_question`, `retrieve_summaries_context_per_question`, `retrieve_book_quotes_context_per_question`) alimenta un nodo de filtrado (`keep_only_relevant_content`). Tras el filtrado, una arista condicional verifica si el contenido destilado está fundamentado en el contexto original. Si lo está, el subgrafo termina y entrega el contexto relevante; si no, se reintenta el filtrado. Esta arquitectura modular permite que el agente principal seleccione dinámicamente qué subgrafo invocar según la naturaleza de cada tarea, optimizando la precisión de la recuperación.

Creamos funciones de recuperación individuales para cada fuente y construimos un subgrafo para cada una:

1. **Subgrafo de chunks del manual:** `retrieve_chunks_context_per_question` → `keep_only_relevant_content` → verificación de destilación.
2. **Subgrafo de resúmenes de capítulos:** `retrieve_summaries_context_per_question` → `keep_only_relevant_content` → verificación de destilación.
3. **Subgrafo de citas normativas:** `retrieve_book_quotes_context_per_question` → `keep_only_relevant_content` → verificación de destilación.

Cada subgrafo tiene un ciclo de verificación: si el contenido destilado no está fundamentado en el original, se reintenta el filtrado.

---

## Creación del Subgrafo para Mitigación de Alucinaciones

![Subgrafo Anti-Alucinaciones](https://miro.medium.com/v2/resize:fit:634/1*EIb0KqHB9_0F9XQV8tULHQ.png)

**Explicación del diagrama en español:** Este subgrafo implementa un ciclo cerrado de verificación de fidelidad. Consta de un único nodo `answer_question_from_context` que genera la respuesta usando razonamiento CoT a partir del contexto agregado. Inmediatamente después, una arista condicional ejecuta `is_answer_grounded_on_context`, que verifica si la respuesta está fundamentada en los hechos del manual. Si se detecta una alucinación (información no soportada por el contexto), el flujo retroalimenta al mismo nodo de respuesta para un nuevo intento, forzando al modelo a ajustarse estrictamente al contenido del Manual de Amaranto. Si la respuesta está fundamentada, el subgrafo termina. Este mecanismo es esencial para el dominio legal/residencial, donde una respuesta incorrecta sobre sanciones u obligaciones podría tener consecuencias reales para los copropietarios.

Creamos un subgrafo dedicado a verificar que las respuestas no contengan alucinaciones. El grafo es cíclico: si la respuesta no está fundamentada en el contexto, se reintenta la generación hasta obtener una respuesta fiel al Manual de Amaranto.

Probamos este subgrafo forzando un contexto limitado (por ejemplo, proporcionando información incorrecta sobre una norma del manual) y verificamos que el sistema responde exclusivamente basado en el contexto, sin inventar información.

---

## Creación y Prueba del Ejecutor de Planes

![Ejecutor de Planes](https://miro.medium.com/v2/resize:fit:2000/1*2g1h5ZkrmC1CWs-igmUD_g.png)

**Explicación del diagrama en español:** El ejecutor de planes es el componente de inteligencia central del pipeline. Opera sobre un estado tipado (`PlanExecute`) que mantiene: la pregunta original, la pregunta anonimizada, el plan actual (lista de pasos), los pasos ya ejecutados, el mapeo de variables anonimizadas, el contexto actual y el contexto agregado acumulado. El planificador utiliza GPT-4o para descomponer una consulta compleja en pasos secuenciales. Por ejemplo, ante la pregunta *"¿Puedo hacer un asado el domingo y qué pasa si mi perro ladra en la noche?"*, el planificador podría generar: (1) buscar normas sobre uso de zonas sociales y parrillas, (2) buscar normas sobre horarios de actividades, (3) buscar normas sobre tenencia de mascotas y ruido, (4) buscar régimen de sanciones aplicable, (5) consolidar la respuesta regulatoria. Cada paso es luego refinado por el desglosador de planes para garantizar que sea ejecutable por una de las herramientas disponibles.

El **ejecutor de planes** es el componente que dota de inteligencia al pipeline. Utiliza GPT-4o para generar un plan paso a paso a partir de la pregunta del usuario:

```python
class Plan(BaseModel):
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")
```

Para una pregunta compleja como *"¿Cuáles son las normas para el uso del salón social y qué sanciones aplican por incumplimiento?"*, el planificador genera pasos como:

1. Buscar información sobre uso del salón social en los fragmentos del manual.
2. Buscar normas específicas sobre reservas y horarios en los resúmenes de capítulos.
3. Buscar sanciones por incumplimiento en las citas normativas.
4. Consolidar la respuesta con base en el contexto recuperado.

El plan luego se **desglosa** en tareas ejecutables por las herramientas del sistema (recuperar chunks, recuperar resúmenes, recuperar citas, responder desde contexto).

---

## Lógica de Re-Planificación

El **Re-Planificador** actualiza el plan dinámicamente basándose en el progreso acumulado. Recibe como entrada:

- La pregunta original.
- El plan actual.
- Los pasos ya ejecutados.
- El contexto agregado hasta el momento.

Si el contexto acumulado es insuficiente, el re-planificador ajusta la estrategia, eliminando pasos completados y añadiendo nuevas tareas. Esto permite que el sistema maneje consultas donde la primera ronda de recuperación no es concluyente, un escenario común en documentos legales donde la información relevante puede estar distribuida en múltiples secciones no adyacentes.

---

## Creación del Manejador de Tareas

El **Manejador de Tareas** decide dinámicamente qué herramienta utilizar para cada paso del plan. Las herramientas disponibles son:

| Herramienta | Descripción | Uso en el dominio Amaranto |
| :--- | :--- | :--- |
| **Tool A:** `retrieve_chunks` | Recupera fragmentos del manual | Búsqueda general de normas y disposiciones |
| **Tool B:** `retrieve_summaries` | Recupera resúmenes de capítulos | Visión general de un tema (ej. "mascotas") |
| **Tool C:** `retrieve_quotes` | Recupera citas normativas textuales | Búsqueda de sanciones específicas o artículos |
| **Tool D:** `answer_from_context` | Responde desde el contexto agregado | Solo cuando la evidencia acumulada es suficiente |

El manejador de tareas recibe como entrada la tarea actual, el contexto agregado, la última herramienta utilizada, los pasos previos y la pregunta original. Con esta información, decide qué herramienta invocar y qué consulta específica formular.

---

## Anonimización/Des-Anonimización de la Pregunta

Para generar un plan sin sesgos basados en el conocimiento previo del LLM, implementamos un ciclo de **anonimización/des-anonimización**:

### Anonimización

Las entidades específicas del dominio residencial se reemplazan por variables:

- *"¿Puede el apartamento 402 hacer una mudanza el sábado?"* → *"¿Puede el apartamento X hacer una mudanza el día Y?"*
- Mapeo: `{"X": "402", "Y": "sábado"}`

Esto evita que el LLM genere un plan sesgado por su conocimiento previo sobre normativas de propiedad horizontal (que pueden variar entre ciudades o conjuntos).

### Des-Anonimización

Una vez generado el plan con variables anonimizadas, se sustituyen las variables por los valores originales, produciendo un plan concreto y ejecutable.

Este ciclo garantiza que la planificación se base exclusivamente en la estructura de la pregunta, no en el conocimiento pre-entrenado del modelo.

---

## Compilación y Visualización del Pipeline RAG Completo

![Agente Finalizado](https://miro.medium.com/v2/resize:fit:2000/1*Jy95B9p_-VERmZMjHxNryw.png)

**Explicación del diagrama en español:** El grafo completo del agente Plan-and-Execute integra todos los componentes en un flujo de estado cíclico con 11 nodos. El punto de entrada es `anonymize_question`, que reemplaza entidades específicas por variables. El flujo avanza secuencialmente por `planner` (generación del plan de alto nivel), `de_anonymize_plan` (restauración de entidades originales) y `break_down_plan` (refinamiento en tareas ejecutables). En `task_handler`, una decisión condicional con cuatro salidas selecciona el subgrafo apropiado: `retrieve_chunks`, `retrieve_summaries`, `retrieve_book_quotes` o `answer`. Los tres nodos de recuperación y el nodo de respuesta confluyen en `replan`, donde otra decisión condicional evalúa si la pregunta original ya puede ser respondida. Si es así, el flujo avanza a `get_final_answer` y termina; si no, retroalimenta a `break_down_plan` para una nueva iteración con el contexto enriquecido. El grafo se visualiza en modo *x-ray* para mostrar la estructura interna de los subgrafos anidados.

El pipeline completo se compila como un grafo de LangGraph con el estado `PlanExecute`. El flujo de alto nivel es:

1. **Anonimizar pregunta** → reemplazar entidades por variables.
2. **Planificar** → generar estrategia de alto nivel.
3. **Des-anonimizar plan** → restaurar entidades originales.
4. **Desglosar plan** → refinar en tareas ejecutables.
5. **Manejador de tareas** → seleccionar herramienta (chunks / summaries / quotes / answer).
6. **Ejecutar herramienta** → recuperar o responder.
7. **Re-planificar** → evaluar si se puede responder o se necesita otra iteración.
8. **Si se puede responder** → generar respuesta final.
9. **Fin** → entregar respuesta verificada.

---

## Prueba del Pipeline Finalizado

Probamos el pipeline con tres tipos de preguntas representativas del dominio Amaranto:

### Ejemplo 1: Pregunta sin información en el manual (prueba de no-alucinación)

Ante una pregunta cuya respuesta no está en el Manual de Convivencia, el sistema itera a través de sus herramientas de búsqueda pero, al no encontrar evidencia, responde con: *"Información no encontrada en el Manual de Convivencia de Amaranto Club House"*. Esto demuestra que el sistema no inventa respuestas cuando la información no está disponible.

### Ejemplo 2: Pregunta compleja que requiere razonamiento multi-paso

Para una consulta como *"¿Qué normas aplican al uso del salón social para eventos familiares y cuáles son las sanciones por incumplimiento?"*, el sistema:

1. Planifica los pasos necesarios.
2. Busca en fragmentos del manual información sobre salón social.
3. Busca en resúmenes de capítulos las normas de uso de zonas comunes.
4. Busca en citas normativas las sanciones aplicables.
5. Consolida una respuesta regulatoria completa citando los capítulos fuente.

### Ejemplo 3: Pregunta que requiere razonamiento CoT

Para una consulta que requiere inferencia lógica (por ejemplo, determinar si una actividad específica está permitida cruzando normas de horarios, zonas y ruido), el sistema muestra su cadena de razonamiento paso a paso antes de emitir la respuesta final.

---

## Evaluación con RAGAS

Para evaluar el pipeline utilizamos **RAGAS**, una librería especializada en la evaluación de aplicaciones LLM. Definimos un conjunto de preguntas sintéticas desde la perspectiva de copropietarios de Amaranto y sus correspondientes *ground truths* (verdades de campo) basadas estrictamente en el Manual de Convivencia.

### Métricas de Evaluación

| Métrica | Descripción | Relevancia para el dominio Amaranto |
| :--- | :--- | :--- |
| **Answer Correctness** | Precisión factual de la respuesta | Crítica: una sanción mal citada tiene implicaciones legales |
| **Faithfulness** | Fidelidad al contexto recuperado | Crítica: la respuesta debe ceñirse al manual, sin invenciones |
| **Answer Relevancy** | Pertinencia de la respuesta a la pregunta | Alta: la respuesta debe abordar exactamente lo preguntado |
| **Context Recall** | Cobertura del contexto relevante | Alta: no debe omitirse información pertinente del manual |
| **Answer Similarity** | Similitud semántica con la verdad de campo | Alta: referencia para comparación con la respuesta esperada |

### Conjunto de Datos de Evaluación (Ejemplos del Dominio Amaranto)

| Pregunta | Verdad de Campo (Ground Truth) |
| :--- | :--- |
| ¿Cuál es el horario permitido para el uso de la piscina? | [Extraído del capítulo de Zonas Sociales del Manual] |
| ¿Qué sanciones aplican por ruido excesivo después de las 10 PM? | [Extraído del Régimen de Sanciones del Manual] |
| ¿Cuál es el procedimiento para solicitar una mudanza? | [Extraído del capítulo de Mudanzas del Manual] |

### Resultados de la Evaluación

El pipeline se evalúa ejecutando cada pregunta, recuperando el contexto, generando la respuesta y comparándola con la verdad de campo usando las cinco métricas. Los resultados se presentan en un DataFrame de pandas para su análisis, permitiendo identificar qué componentes del pipeline requieren ajustes (por ejemplo, si *faithfulness* es bajo, se debe reforzar el fact-checker; si *context recall* es bajo, se debe ajustar la recuperación vectorial).

---

## Resumen General

Hemos construido, desde cero, un sistema RAG de nivel producción aplicado al dominio del Manual de Convivencia del Conjunto Residencial Amaranto Club House. El proceso abarcó:

1. **Preprocesamiento y limpieza de datos:** normalización de texto del PDF del manual.
2. **Fragmentación híbrida:** división tradicional (chunks con solapamiento) y lógica (capítulos, citas normativas).
3. **Vectorización:** creación de tres índices FAISS con embeddings de OpenAI.
4. **Pipeline central:** recuperador multi-fuente, filtro de relevancia, reescritor de consultas, razonamiento CoT y verificación de fidelidad.
5. **Enfoque de subgrafos:** módulos especializados para cada tipo de recuperación y para mitigación de alucinaciones.
6. **Agente Plan-and-Execute:** planificador, desglosador de tareas, manejador de herramientas, re-planificador y ciclo de anonimización/des-anonimización.
7. **Visualización con LangGraph:** grafos de estado cíclicos que muestran el flujo completo.
8. **Evaluación con RAGAS:** cinco métricas para medir la calidad del sistema en el dominio legal/residencial.

### Lecciones Aprendidas

- Un sistema RAG robusto para documentos legales/residenciales requiere **múltiples índices vectoriales** que capturen diferentes granularidades de la información (chunks para contexto amplio, resúmenes para visión general, citas para evidencia textual específica).
- La **planificación dinámica** con re-planificación es esencial cuando la información relevante está dispersa en múltiples secciones no adyacentes del documento.
- El **blindaje anti-alucinaciones** es crítico en dominios donde las respuestas tienen implicaciones legales o regulatorias. El fact-checker debe ser un componente obligatorio, no opcional.
- La **evaluación sistemática** con RAGAS permite cuantificar la calidad del pipeline y dirigir los esfuerzos de mejora hacia los componentes con métricas más bajas.
- La **anonimización de preguntas** reduce sesgos del LLM al generar planes, forzando al sistema a basarse en la estructura de la consulta y no en conocimiento pre-entrenado que podría no aplicar al dominio específico de Amaranto.

---

Basado en el repositorio: [https://github.com/FareedKhan-dev/complex-RAG-guide](https://github.com/FareedKhan-dev/complex-RAG-guide)

Adaptado al dominio del Manual de Convivencia del Conjunto Residencial Amaranto Club House por Fredy Sarmiento, Manuel Caro y Jorge Bravo — Universidad del Rosario, Maestría ICT.