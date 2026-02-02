import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.table import Table
from typing import Optional, Dict, List
import time


console = Console()


class RemoteSyncManager:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.tenant = tenant
        self.database = database
        self.api_key = api_key
        self.verbose = verbose
        self.client = None

    def connect(self) -> bool:
        try:
            settings = Settings()

            headers = None
            if self.api_key:
                headers = {"x-chroma-token": self.api_key}

            if self.tenant or self.database:
                kwargs = {
                    "host": self.host,
                    "port": self.port,
                    "headers": headers,
                }
                if self.tenant:
                    kwargs["tenant"] = self.tenant
                if self.database:
                    kwargs["database"] = self.database
                self.client = chromadb.HttpClient(**kwargs)
            else:
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    headers=headers,
                    settings=settings
                )

            if self.verbose:
                console.print(f"✓ Connected to ChromaDB at http://{self.host}:{self.port}")

            return True
        except Exception as e:
            console.print(f"❌ Failed to connect to remote ChromaDB: {e}", style="red")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self.client.list_collections()
            if self.verbose:
                existing_names = [coll.name for coll in collections]
                console.print(f"  Collections on remote: {existing_names}")
            exists = any(coll.name == collection_name for coll in collections)
            return exists
        except Exception as e:
            if self.verbose:
                console.print(f"Error checking collection existence: {e}", style="yellow")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            return {
                "name": collection_name,
                "count": count,
            }
        except Exception as e:
            if self.verbose:
                console.print(f"Error getting collection info: {e}", style="yellow")
            return None

    def handle_conflict(self, collection_name: str, local_data: Dict) -> str:
        console.print(f"\n⚠️  Collection '{collection_name}' already exists on remote!", style="yellow")

        remote_info = self.get_collection_info(collection_name)
        local_count = len(local_data.get("ids", []))
        remote_count = remote_info.get("count", 0) if remote_info else 0

        table = Table(title="Collection Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Collection Name", collection_name)
        table.add_row("Local Chunks", str(local_count))
        table.add_row("Remote Chunks", str(remote_count))

        console.print(table)

        console.print("\nHow would you like to proceed?", style="bold")
        console.print("1. Skip - Do not sync this collection", style="white")
        console.print("2. Overwrite - Replace remote collection with local data", style="white")
        console.print("3. Merge - Merge local data into remote collection (may create duplicates)", style="white")

        while True:
            choice = console.input("\n[bold]Enter choice (1/2/3): [/bold]").strip()

            if choice == "1":
                return "skip"
            elif choice == "2":
                confirm = console.input(
                    f"[red]Are you sure you want to overwrite '{collection_name}' on remote? (y/N): [/red]"
                ).strip().lower()
                if confirm == "y":
                    return "overwrite"
                console.print("Aborted overwrite.", style="yellow")
                return "skip"
            elif choice == "3":
                confirm = console.input(
                    f"[yellow]Continue merging into '{collection_name}'? (y/N): [/yellow]"
                ).strip().lower()
                if confirm == "y":
                    return "merge"
                console.print("Aborted merge.", style="yellow")
                return "skip"
            else:
                console.print("Invalid choice. Please enter 1, 2, or 3.", style="red")

    def sync_collection(
        self,
        collection_name: str,
        local_data: Dict,
        metadata: Optional[Dict] = None,
        reset_remote: bool = False,
    ) -> bool:
        if not self.client:
            if not self.connect():
                return False

        collection_exists = self.collection_exists(collection_name)

        if collection_exists:
            if reset_remote:
                self.client.delete_collection(collection_name)
                console.print(f"🗑️  Reset: deleted old collection '{collection_name}'", style="yellow")
                
                max_retries = 10
                for attempt in range(max_retries):
                    time.sleep(0.5)
                    if not self.collection_exists(collection_name):
                        if self.verbose:
                            console.print(f"  Confirmed deletion after {attempt + 1} attempts")
                        break
                    if attempt < max_retries - 1 and self.verbose:
                        console.print(f"  Waiting for deletion... (attempt {attempt + 1}/{max_retries})")
                
                if self.collection_exists(collection_name):
                    console.print(f"⚠️  Warning: Collection '{collection_name}' still exists after deletion attempts", style="yellow")
                
                collection_exists = False
                action = "overwrite"
            else:
                action = self.handle_conflict(collection_name, local_data)

            if action == "skip":
                console.print(f"⏭️  Skipping sync for collection '{collection_name}'", style="yellow")
                return False
            elif action == "overwrite":
                self.client.delete_collection(collection_name)
                console.print(f"🗑️  Deleted old collection '{collection_name}'", style="yellow")
            elif action == "merge":
                console.print(f"🔀 Merging data into existing collection '{collection_name}'", style="yellow")

        try:
            collection_metadata = metadata or {
                "description": f"ChromaDB collection for {collection_name}"
            }

            if collection_metadata:
                collection_metadata = {k: v for k, v in collection_metadata.items() if v is not None}

            if not collection_exists or action == "overwrite":
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata=collection_metadata
                )
                console.print(f"✓ Created collection '{collection_name}' on remote", style="green")
            else:
                collection = self.client.get_collection(collection_name)

            ids = local_data["ids"]
            documents = local_data.get("documents", [])
            metadatas = local_data.get("metadatas", [])
            embeddings = local_data.get("embeddings", [])

            batch_size = 1000
            total_added = 0

            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                
                if len(documents) > 0:
                    batch_docs = documents[i:i + batch_size]
                else:
                    batch_docs = None
                    
                if len(metadatas) > 0:
                    batch_metas = metadatas[i:i + batch_size]
                else:
                    batch_metas = None
                    
                if len(embeddings) > 0:
                    batch_embeds = embeddings[i:i + batch_size]
                    batch_embeds_fixed = [
                        list(e) if hasattr(e, 'tolist') else 
                        list(e) if isinstance(e, (list, tuple)) else 
                        e for e in batch_embeds
                    ]
                else:
                    batch_embeds_fixed = None

                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeds_fixed
                )

                total_added += len(batch_ids)
                if self.verbose:
                    console.print(f"  Added batch {i // batch_size + 1}: {len(batch_ids)} chunks")

            console.print(f"✅ Successfully synced {total_added} chunks to '{collection_name}' on remote", style="bold green")

            return True

        except Exception as e:
            console.print(f"❌ Error syncing collection '{collection_name}': {e}", style="red")
            return False

    def delete_all_collections(self) -> int:
        if not self.client:
            if not self.connect():
                return 0

        try:
            collections = self.client.list_collections()
            deleted_count = 0

            for coll in collections:
                self.client.delete_collection(coll.name)
                if self.verbose:
                    console.print(f"  🗑️  Deleted collection '{coll.name}'", style="yellow")
                deleted_count += 1

            console.print(f"🗑️  Deleted {deleted_count} collection(s) on remote", style="red")
            return deleted_count

        except Exception as e:
            console.print(f"❌ Error deleting collections: {e}", style="red")
            return 0

    def delete_collection(self, collection_name: str) -> bool:
        if not self.client:
            if not self.connect():
                return False

        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                console.print(f"🗑️  Deleted collection '{collection_name}' on remote", style="red")
                return True
            else:
                console.print(f"⚠️  Collection '{collection_name}' does not exist on remote", style="yellow")
                return False
        except Exception as e:
            console.print(f"❌ Error deleting collection '{collection_name}': {e}", style="red")
            return False

    def close(self):
        self.client = None