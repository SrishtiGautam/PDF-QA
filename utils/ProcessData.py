import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class ProcessPDF:
    def __init__(self, cfg, logger, lm_agent):
        self.cfg = cfg
        self.logger = logger
        self.path = cfg.pdf_path
        self.chunks = []
        self.embeddings = []
        self.LMAgent = lm_agent

    def process(self):
        with open(self.path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page in tqdm(range(len(pdf_reader.pages)), desc="Processing Pages", unit="page"):
                text = pdf_reader.pages[page].extract_text()
                for i in range(0, len(text), self.cfg.chunk_size - self.cfg.chunk_overlap):
                    chunk = text[i:i + self.cfg.chunk_size]
                    self.chunks.append(chunk)
                    if self.cfg.embed:
                        self.embeddings.append(self.LMAgent.embed_chunk(chunk))



class RetrievalAgent:
    def __init__(self, lm_agent, pdf_agent):
        self.LMAgent = lm_agent
        self.PDFAgent = pdf_agent

    def find_relevant_context(self, query, top_k=3):
        # Generate embedding for the query
        query_embedding = self.LMAgent.embed_chunk(query)

        # Calculate cosine similarity between query embedding and context embeddings
        similarities = cosine_similarity([query_embedding], self.PDFAgent.embeddings)[0]

        # Get the indices of the top_k most similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.PDFAgent.chunks[i] for i in top_indices]
