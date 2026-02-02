from typing import List
from sentence_transformers import SentenceTransformer
import rich.progress as progress


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", verbose: bool = False, use_cpu: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.use_cpu = use_cpu
        self.model = None

    def load_model(self):
        if self.verbose:
            device_str = "CPU" if self.use_cpu else "GPU"
            print(f"Loading embedding model on {device_str}: {self.model_name}")

        device = "cpu" if self.use_cpu else None
        self.model = SentenceTransformer(self.model_name, device=device)
        return self.model

    def get_dimension(self) -> int:
        if not self.model:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        if not self.model:
            self.load_model()

        if show_progress and self.verbose:
            with progress.Progress(
                "[progress.description]{task.description}",
                progress.BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                progress.TimeRemainingColumn(),
            ) as pbar:
                task = pbar.add_task(
                    f"[cyan]Embedding {len(texts)} chunks...",
                    total=len(texts)
                )
                embeddings = []

                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    embeddings.extend(batch_embeddings.tolist())
                    pbar.update(task, advance=len(batch))

                return embeddings
        else:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)