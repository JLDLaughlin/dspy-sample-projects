import numpy as np
import tiktoken

CHUNK_SIZE = 300  # tokens
CHUNK_OVERLAP = 50


def load_chunks(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_tokens = tokens[i : i + CHUNK_SIZE]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


def get_embedding(client, text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    # Ensure the embedding is a numpy array of floats
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_rag(client, query):
    # Load and embed chunks from knowledge base
    chunks = load_chunks("knowledge.md")
    chunk_embeddings = [get_embedding(client, c) for c in chunks]

    # Embed the user query
    query_embedding = get_embedding(client, query)

    # Rank by similarity
    scored = [
        (cosine_similarity(query_embedding, emb), chunk)
        for chunk, emb in zip(chunks, chunk_embeddings)
    ]
    top_chunks = [c for _, c in sorted(scored, reverse=True)[:3]]

    # Inject into GPT prompt
    messages = [
        {
            "role": "system",
            "content": "Answer using the following context:\n"
            + "\n---\n".join(top_chunks),
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.3
    )

    return response.choices[0].message.content


def rewrite_query(client, query):
    messages = [
        {
            "role": "system",
            "content": "Rewrite the user query to better match internal documentation.",
        },
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0
    )
    return response.choices[0].message.content.strip()


def refine_response(client, raw_answer):
    messages = [
        {
            "role": "system",
            "content": "Rewrite this customer service response to be shorter, polite, and professional.",
        },
        {"role": "user", "content": raw_answer},
    ]
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.2
    )
    return response.choices[0].message.content.strip()
