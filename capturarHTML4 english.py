import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from elasticsearch import Elasticsearch
import time
import urllib3
import logging

nltk.download('punkt')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Cargar el modelo pre-entrenado para los embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Conectar a Elasticsearch (ajusta la configuración si es necesario)
es = Elasticsearch(
    ["https://127.0.0.1:9200"],
    verify_certs=False,
    basic_auth=("elastic", "nTFVUgiWvriLj-eYJvsh"),
)


def extract_text_from_html(html_content):
    """Extrae texto, segmenta y genera embeddings."""

    soup = BeautifulSoup(html_content, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))
    sentence_embeddings = model.encode(sentences)
    return text, paragraphs, sentences, sentence_embeddings


def create_faiss_index(embeddings):
    """Crea un índice FAISS."""

    dimension = embeddings.shape[1]
    nlist = min(10, len(embeddings))  # Ajuste dinámico de nlist
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(embeddings)
    index.add(embeddings)
    return index


def search_faiss_index(index, query_embedding, k=10):
    """Busca en el índice FAISS."""

    distances, indices = index.search(np.array([query_embedding]), k)
    return distances[0], indices[0]


def create_elasticsearch_index(index_name, documents):
    """Crea un índice en Elasticsearch."""

    if not es.indices.exists(index=index_name):
        mapping = {"properties": {"content": {"type": "text"}}}
        es.indices.create(index=index_name, mappings=mapping)
    for i, doc in enumerate(documents):
        es.index(index=index_name, id=i, document={"content": doc})


def search_elasticsearch_index(index_name, query, size=10):
    """Busca en Elasticsearch."""

    body = {"query": {"match": {"content": query}}, "size": size}
    res = es.search(index=index_name, body=body)
    return res['hits']['hits']


def process_urls(urls):
    """Procesa una lista de URLs."""

    all_text = []
    all_paragraphs = []
    all_sentences = []
    all_sentence_embeddings = []

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.content
            text, paragraphs, sentences, sentence_embeddings = extract_text_from_html(
                html_content
            )
            all_text.append(text)
            all_paragraphs.extend(paragraphs)
            all_sentences.extend(sentences)
            all_sentence_embeddings.extend(sentence_embeddings)
            print(f"Procesando: {url}")
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"Error al procesar {url}: {e}")
            continue

    return all_text, all_paragraphs, all_sentences, all_sentence_embeddings


# 1. Definir las URLs a procesar
urls = ["https://ipt.gbif.org/manual/en/ipt/latest/overview",
        "https://ipt.gbif.org/manual/en/ipt/latest/home"]
    # ... Agrega todas las URLs que necesites procesar


# 2. Extraer datos y generar embeddings
(
    all_text,
    all_paragraphs,
    all_sentences,
    all_sentence_embeddings,
) = process_urls(urls)

# Print paragraph and sentence counts
print(f"\nTotal paragraphs extracted: {len(all_paragraphs)}")
print(f"Total sentences extracted: {len(all_sentences)}\n")

# 3. Crear índices de búsqueda
# Crear índice FAISS
faiss_index = create_faiss_index(np.array(all_sentence_embeddings))

# Crear índice de Elasticsearch (usando los párrafos como ejemplo)
create_elasticsearch_index("mi_indice", all_paragraphs)


# 4. Implementar la función de búsqueda combinada (ejemplo)
def combined_search(
    query,
    faiss_index,
    es_index_name,
    model,
    all_sentences,
    all_paragraphs,
    alpha=0.5,
):
    """Realiza una búsqueda combinada FAISS + Elasticsearch con ponderación variable."""
    # Obtener embedding de la consulta
    query_embedding = model.encode(query)

    # FAISS (distancias)
    distances, faiss_indices = search_faiss_index(faiss_index, query_embedding, k=5)
    faiss_results = []
    for i, idx in enumerate(faiss_indices):
        sim_score = 1 - distances[i]  # convertir distancia a similitud
        faiss_results.append({"text": all_sentences[idx], "score": sim_score, "source": "faiss"})

    # Elasticsearch
    es_raw = search_elasticsearch_index(es_index_name, query, size=5)
    es_results = []
    for hit in es_raw:
        es_results.append({"text": hit["_source"]["content"], "score": hit["_score"], "source": "es"})

    # Normalizar scores
    max_faiss = max(r["score"] for r in faiss_results) if faiss_results else 1
    max_es = max(r["score"] for r in es_results) if es_results else 1

    for r in faiss_results:
        r["score"] = (r["score"] / max_faiss) if max_faiss > 0 else 0
    for r in es_results:
        r["score"] = (r["score"] / max_es) if max_es > 0 else 0

    # Fusionar
    combined = []
    texts_seen = set()
    for r in faiss_results + es_results:
        if r["text"] not in texts_seen:
            texts_seen.add(r["text"])
            combined.append(r)

    # Asignar score combinado
    for r in combined:
        if r["source"] == "faiss":
            r["combined_score"] = alpha * r["score"]
        else:
            r["combined_score"] = (1 - alpha) * r["score"]

    # Ordenar
    combined.sort(key=lambda x: x["combined_score"], reverse=True)

    return combined


# Ejemplo de búsqueda combinada
query = "configure resources"
resultados = combined_search(query, faiss_index, "mi_indice", model, all_sentences, all_paragraphs)

print("\nCombined Search Results:")
for r in resultados:
    print(f"Score: {r['combined_score']:.4f} | Source: {r['source']} | Text: {r['text'][:80]}...")


test_queries = [
    "configure resources",
    "delete published versions",
    "view data inventory",
    "upload a new dataset",
    "how to sort data"
]

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"\n🔎 Alpha: {alpha}")
    for query in test_queries:
        print(f"\n🟡 Query: {query}")
        resultados = combined_search(query, faiss_index, "mi_indice", model, all_sentences, all_paragraphs, alpha=alpha)
        for r in resultados[:3]:  # mostrar top 3
            print(f"Score: {r['combined_score']:.4f} | Source: {r['source']} | Text: {r['text'][:80]}...")


def precision_at_k(results, relevant_texts, k=5):
    hits = sum(1 for r in results[:k] if any(rel in r['text'] for rel in relevant_texts))
    return hits / k


# Determinar relevancia (debes hacer esto manualmente)
relevant_texts = {
    "To learn more, see the information in the section \"Editing an Existing Resource\" in the \"Resource Management\" menu.",
    "The next step is to proceed with configuring each shared resource."
}  # Ajusta esto a tus documentos relevantes


precision = precision_at_k(resultados, relevant_texts)
print(f"\nPrecision@5: {precision:.4f}")