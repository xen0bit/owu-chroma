import hashlib
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class LocalDBManager:
    def __init__(self, db_path: str, db_name: str, verbose: bool = False):
        self.db_path = Path(db_path)
        self.db_name = db_name
        self.verbose = verbose
        self.client = None
        self.collection = None

    def create_database(self) -> bool:
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            self.collection = self.client.get_or_create_collection(
                name=self.db_name,
                metadata={
                    "description": f"ChromaDB for {self.db_name}"
                }
            )

            if self.verbose:
                print(f"Created ChromaDB at: {self.db_path}")
                print(f"Collection name: {self.db_name}")

            return True

        except Exception as e:
            print(f"Error creating database: {e}")
            return False

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        if not self.collection:
            if self.verbose:
                print("Collection not initialized")
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            content_hash = hashlib.md5(
                chunk["content"].encode('utf-8')
            ).hexdigest()[:16]
            unique_id = f"{chunk['source_file'].replace('/', '_')}_{content_hash}"

            ids.append(unique_id)
            documents.append(chunk["content"])
            metadatas.append({
                "source_file": chunk["source_file"],
                **chunk["metadata"],
            })

        try:
            batch_size = 1000
            total_added = 0

            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_embeds = embeddings[i:i + batch_size]

                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeds
                )

                total_added += len(batch_ids)
                if self.verbose:
                    print(f"Added batch {i // batch_size + 1}: {len(batch_ids)} chunks")

            if self.verbose:
                print(f"Total chunks added: {total_added}")

            return total_added

        except Exception as e:
            print(f"Error adding chunks: {e}")
            return 0

    def get_stats(self) -> Dict:
        if not self.collection:
            return {}

        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.db_name,
            }
        except Exception as e:
            if self.verbose:
                print(f"Error getting stats: {e}")
            return {}

    def get_all_chunks(self) -> Dict:
        if not self.collection:
            return {}

        try:
            results = self.collection.get(
                include=["metadatas", "documents", "embeddings"]
            )
            return results
        except Exception as e:
            if self.verbose:
                print(f"Error getting chunks: {e}")
            return {}

    def close(self):
        self.client = None
        self.collection = None

    def get_collection(self):
        return self.collection

    def get_client(self):
        return self.client