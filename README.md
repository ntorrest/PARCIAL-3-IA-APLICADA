
# PARCIAL-3-IA-APLICADA

Integrantes: 

En este proyecto se desarrolló un sistema RAG para responder preguntas sobre el conflicto armado colombiano, mediante información del informe de la comisión de la verdad “Basta ya Colombia: Memorias de Guerra y Dignidad” que cuenta con alrededor de 430 páginas. Se escogió este documento ya que brinda un contexto general sobre todo lo ocurrido en la época del conflicto armado, por lo que puede esperarse que cualquier pregunta concreta que se tenga sobre este tema se encuentre en el documento. Esta es precisamente la intención que tenemos de este RAG, aumentar la posibilidad de obtener una misma versión ante una pregunta sobre esta época, para no tener que recurrir a muchas fuentes o perderse en la extensión del documento utilizado.  


## 1. Extracción y limpieza del texto

El primer paso consiste en leer el PDF y sacar el texto página por página. Para esto se uso `pypdf`. El código también tiene un manejo básico de errores por si alguna página viene vacía o con problemas. El objetivo aquí es simplemente obtener el texto limpio con su número de página.

```python
def load_pdf_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text.strip()))
    return pages

```

## 2. Chunking del contenido
Como los modelos de lenguaje y los índices vectoriales no trabajan bien con textos demasiado largos, el PDF se divide en fragmentos más manejables. En este caso se usaron chunks de 1200 caracteres y, para no perder continuidad, dejo un solapamiento de 200. Además, guardo algunos datos como la página, la posición del texto y un ID único por chunk (metadata). Esto sirve mucho después cuando uno quiere saber de dónde salió un fragmento.

```python
def chunk_page_text(page_num, text, pdf_name):
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if not chunk_text:
            break

        obj = {
            "pdf_name": pdf_name,
            "page": page_num,
            "chunk_id": f"{pdf_name}_p{page_num}_c{chunk_idx}",
            "text": chunk_text,
            "char_start": start,
            "char_end": min(end, len(text)),
            "position": chunk_idx
        }

        chunks.append(obj)
        chunk_idx += 1
        start = end - CHUNK_OVERLAP

    return chunks

```

## 3. Embeddings con BGE-M3
Después de tener los chunks listos, lo siguiente es convertirlos en vectores. Para esto uso el modelo BGE-M3, que es multilingüe y funciona muy bien para tareas de búsqueda semántica. Los embeddings se normalizan porque eso ayuda a que FAISS encuentre coincidencias de manera más estable.

```python
self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

texts = [c["text"] for c in self.corpus]

self.embeddings = self.model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)
self.embeddings = np.array(self.embeddings).astype("float32")

```

## 4. Índice vectorial con FAISS
Con los embeddings generados, se guardaron en un índice FAISS. Se eligio indice IndexFlatL2 porque es sencillo y da resultados exactos (no aproximados).

```python

dim = self.embeddings.shape[1]
self.index = faiss.IndexFlatL2(dim)
self.index.add(self.embeddings)

```

## 5. Modelo generativo
En este archivo todavía no se uso un modelo generativo directamente, pero el sistema está pensado para integrarse fácilmente con uno (como GPT-3.5 o GPT-4). Los resultados del buscador están listos para ser incluidos como contexto en un prompt RAG.

## 6. Construcción del prompt RAG

La idea es sencilla: tomar los fragmentos recuperados por FAISS y usarlos como contexto para generar una respuesta con base al modelo generativo, que en este caso es la API de chatgpt

## 7. Respuesta final
La respuesta final dependerá del modelo generativo que se conecte al sistema, como se menciono anteriormente, esta conectado a la API de ChatGPT. Con esto se genera una respuesta que muestra los documentos más relevantes para la busqueda, adicional de una respuesta generada:


```python
    if __name__ == "__main__":
    searcher = PDFSemanticSearcher(PDF_PATH)

    print("Buscador cargado. Pregunta lo que quieras del PDF.\n")

    while True:
        query = input("Pregunta: ").strip()
        if query.lower() in ["salir", "exit", "quit"]:
            print("Chao 👋")
            break

        results = searcher.search(query, top_k=5)

        print("\n========= TOP K RESULTADOS =========\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.4f}")
            print(f"PDF: {r['pdf']}")
            print(f"Página: {r['page']} | Chunk: {r['chunk_id']}")
            print(f"Rango caracteres: {r['char_range']}")
            print(f"Posición en página: {r['position']}")
            print("-" * 60)

            preview = r["text"]
            if len(preview) > 700:
                preview = preview[:700] + "..."
            print(preview)
            print("\n" + "="*80 + "\n")

```

## 8. Agente interactivo
Finalmente, el código tiene un bucle que permite hacer preguntas directamente por consola. Cada vez que el usuario escribe una consulta, el sistema la convierte en embedding, busca los fragmentos más parecidos y los muestra en pantalla.

```python

while True:
    query = input("Pregunta: ").strip()
    if query.lower() in ["salir", "exit", "quit"]:
        break

    results = searcher.search(query, top_k=5)

    for r in results:
        print(r["text"])
```
