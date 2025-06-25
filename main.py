import os
import time

import dspy
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from utils import cosine_similarity

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# -------------------
# DSPy Signatures
# -------------------
class AnswerWithContext(dspy.Signature):
    context = dspy.InputField()
    query = dspy.InputField()
    answer = dspy.OutputField()


class RewriteQuery(dspy.Signature):
    original = dspy.InputField()
    rewritten = dspy.OutputField()


# -------------------
# DSPy Modules
# -------------------
class Embedder(dspy.Module):
    def __init__(self, api_key, model="text-embedding-3-small"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy client creation to avoid serialization issues."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def forward(self, text):
        # Create client fresh each time to avoid deep copy issues
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(input=[text], model=self.model)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def __getstate__(self):
        """Custom serialization - exclude the client object."""
        state = self.__dict__.copy()
        state['_client'] = None  # Don't serialize the client
        return state

    def __setstate__(self, state):
        """Custom deserialization - client will be recreated on first use."""
        self.__dict__.update(state)
        self._client = None


class KnowledgeBase(dspy.Module):
    def __init__(self, api_key, model="text-embedding-3-small", path="knowledge.md"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.path = path
        
        with open(path) as f:
            self.chunks = [line.strip() for line in f if line.strip()]
        
        # Pre-compute embeddings using a temporary embedder
        temp_embedder = Embedder(api_key, model)
        self.embeddings = [temp_embedder(chunk) for chunk in self.chunks]

    def retrieve(self, query_embedding, k=3):
        scored = [
            (cosine_similarity(query_embedding, emb), chunk)
            for emb, chunk in zip(self.embeddings, self.chunks)
        ]
        top_chunks = [c for _, c in sorted(scored, reverse=True)[:k]]
        return "\n---\n".join(top_chunks)


class Retriever(dspy.Module):
    def __init__(self, kb, api_key, model="text-embedding-3-small"):
        super().__init__()
        self.kb = kb
        self.api_key = api_key
        self.model = model
        self._embedder = None

    @property
    def embedder(self):
        """Lazy embedder creation to avoid serialization issues."""
        if self._embedder is None:
            self._embedder = Embedder(self.api_key, self.model)
        return self._embedder

    def forward(self, query):
        # Create embedder fresh each time
        embedder = Embedder(self.api_key, self.model)
        query_embedding = embedder(query)
        return self.kb.retrieve(query_embedding)

    def __getstate__(self):
        """Custom serialization - exclude the embedder object."""
        state = self.__dict__.copy()
        state['_embedder'] = None
        return state

    def __setstate__(self, state):
        """Custom deserialization - embedder will be recreated on first use."""
        self.__dict__.update(state)
        self._embedder = None


class QueryRewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewriter = dspy.Predict(RewriteQuery)

    def forward(self, query):
        result = self.rewriter(original=query)
        return result.rewritten


class Answerer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(AnswerWithContext)

    def forward(self, context, query):
        return self.generate(context=context, query=query).answer


class RAGPipeline(dspy.Module):
    def __init__(self, rewriter, retriever, answerer):
        super().__init__()
        self.rewriter = rewriter
        self.retriever = retriever
        self.answerer = answerer

    def forward(self, query, context=None):
        rewritten_query = self.rewriter(query)
        context = self.retriever(rewritten_query)
        answer = self.answerer(context=context, query=rewritten_query)
        return dspy.Prediction(context=context, query=query, answer=answer)


# -------------------
# Pipeline Setup and Main
# -------------------
def dspy_generate_pipeline():
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=client.api_key))

    embedder = Embedder(client.api_key)
    # start = time.time()
    kb = KnowledgeBase(client.api_key, embedder.model)
    # print(f"Initialized KnowledgeBase in {time.time() - start:.2f} seconds")
    rewriter = QueryRewriter()
    retriever = Retriever(kb, client.api_key, embedder.model)
    answerer = Answerer()
    pipeline = RAGPipeline(rewriter, retriever, answerer)

    return pipeline


if __name__ == "__main__":
    # Sample input
    # query = "Hi, I tried to reset my password but never received the email."
    # print("Customer Message:", query)

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=client.api_key))

    start = time.time()
    pipeline = dspy_generate_pipeline()
    print(f"Generated DSPy pipeline in {time.time() - start:.2f} seconds")

    # Interactive mode
    while True:
        query = input("\n📥 Enter a customer message (or type 'exit' to quit):\n> ")
        if query.lower() in ["exit", "quit"]:
            break

        start = time.time()
        answer = pipeline(query)
        print(f"Pipeline ran in {time.time() - start:.2f} seconds")

        print("\nAnswer:\n", answer)

