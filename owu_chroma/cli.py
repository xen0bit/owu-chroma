import zipfile
import sys
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import Config
from .chunkers import get_chunker
from .embedder import Embedder
from .db_manager import LocalDBManager
from .sync_manager import RemoteSyncManager

app = typer.Typer(invoke_without_command=True, no_args_is_help=False, add_completion=False, help="Create and sync ChromaDB from ZIP files")
console = Console()


@app.command(no_args_is_help=False)
def main(
    zip_file: Path = typer.Argument(
        ...,
        help="Path to the ZIP file to process",
        exists=True,
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Database name (default: derived from zip filename)",
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        "-s",
        help="Chunk size in characters (default: 1000)",
    ),
    chunk_overlap: int = typer.Option(
        100,
        "--chunk-overlap",
        "-o",
        help="Chunk overlap in characters (default: 100)",
    ),
    model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--model",
        "-m",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory path (default: current directory)",
    ),
    remote_host: str = typer.Option(
        "127.0.0.1",
        "--remote-host",
        help="Remote ChromaDB host (default: 127.0.0.1)",
    ),
    remote_port: int = typer.Option(
        8080,
        "--remote-port",
        help="Remote ChromaDB port (default: 8080)",
    ),
    remote_tenant: Optional[str] = typer.Option(
        None,
        "--remote-tenant",
        help="Remote ChromaDB tenant (optional)",
    ),
    remote_database: Optional[str] = typer.Option(
        None,
        "--remote-database",
        help="Remote ChromaDB database (optional)",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for remote ChromaDB (optional)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress",
    ),
    reset_remote: bool = typer.Option(
        False,
        "--reset-remote",
        "-r",
        help="Reset remote collection (delete and recreate)",
    ),
    reset_all_remote: bool = typer.Option(
        False,
        "--reset-all",
        "-R",
        help="Delete ALL collections on remote server before syncing",
    ),
    use_cpu: bool = typer.Option(
        False,
        "--cpu",
        "-c",
        help="Force CPU usage for embeddings (default: auto-detect)",
    ),
    remote_only: bool = typer.Option(
        False,
        "--remote-only",
        help="Insert directly to remote without creating local database",
    ),
):
    """
    Create a ChromaDB vector database from a ZIP file and sync to remote server.

    The ZIP file will be extracted, all files will be chunked based on their type,
    and embeddings will be created using the specified model. The resulting database
    will be stored locally and automatically synced to the remote ChromaDB server.

    Use --remote-only to skip local database creation and insert directly to remote.

    Example:
        owu-chroma myproject.zip --verbose --remote-host 192.168.1.100
        owu-chroma myproject.zip --remote-only --verbose
    """
    db_name = name or zip_file.stem

    config = Config(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=model,
        output_path=output,
        db_name=db_name,
        verbose=verbose,
        remote_host=remote_host,
        remote_port=remote_port,
        remote_tenant=remote_tenant,
        remote_database=remote_database,
        api_key=api_key,
    )

    console.print(f"📦 Processing: {zip_file.name}")
    console.print(f"📝 Database: {config.get_db_name()}")
    console.print(f"🔢 Chunk size: {config.chunk_size}")
    console.print(f"🔀 Overlap: {config.chunk_overlap}")
    console.print(f"🤖 Model: {config.embedding_model}")
    console.print(f"🌐 Remote: {config.get_remote_url()}")
    console.print()

    try:
        process_zip(zip_file, config, use_cpu, reset_remote, reset_all_remote, remote_only)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        sys.exit(1)


def prepare_data_structure(chunks: List[Dict], embeddings: List[List[float]]) -> Dict:
    """Prepare the data structure needed for ChromaDB insertion.

    This mimics what LocalDBManager.get_all_chunks() returns.
    """
    ids = []
    documents = []
    metadatas = []
    chunk_counter = {}

    for idx, chunk in enumerate(chunks):
        content_hash = hashlib.md5(
            chunk["content"].encode('utf-8')
        ).hexdigest()[:16]

        source_file = chunk['source_file'].replace('/', '_')

        if source_file not in chunk_counter:
            chunk_counter[source_file] = 0
        chunk_counter[source_file] += 1

        unique_id = f"{source_file}_{content_hash}_{chunk_counter[source_file]}"

        ids.append(unique_id)
        documents.append(chunk["content"])
        metadatas.append({
            "source_file": chunk["source_file"],
            **chunk["metadata"],
        })

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }


def process_zip(zip_path: Path, config: Config, use_cpu: bool = False, reset_remote: bool = False, reset_all_remote: bool = False, remote_only: bool = False):
    all_chunks = []

    console.print("📂 Extracting and chunking files...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files...", total=None)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len([f for f in file_list if not f.endswith('/')])
            progress.update(task, total=total_files)

            for file_info in zip_ref.infolist():
                if file_info.is_dir():
                    continue

                file_path = file_info.filename

                if any(part.startswith('.') for part in file_path.split('/')):
                    if config.verbose:
                        console.print(f"  Skipping: {file_path}")
                    continue

                try:
                    content = zip_ref.read(file_info).decode('utf-8', errors='ignore')

                    if not content.strip():
                        if config.verbose:
                            console.print(f"  Skipping empty: {file_path}")
                        continue

                    chunker = get_chunker(
                        file_path,
                        config.chunk_size,
                        config.chunk_overlap
                    )

                    chunks = chunker.chunk(content, file_path)
                    all_chunks.extend(chunks)

                    if config.verbose:
                        console.print(
                            f"  ✓ {file_path}: {len(chunks)} chunks"
                        )

                    progress.update(task, advance=1)

                except Exception as e:
                    if config.verbose:
                        console.print(f"  ⚠ Error processing {file_path}: {e}")
                    continue

    console.print()

    total_chunks = len(all_chunks)
    console.print(f"✅ Processed {total_files} files")
    console.print(f"✨ Created {total_chunks} chunks")
    console.print()

    if total_chunks == 0:
        console.print("❌ No chunks created. Exiting.", style="red")
        sys.exit(1)

    console.print("🧠 Creating embeddings...")
    embedder = Embedder(config.embedding_model, config.verbose, use_cpu=use_cpu)
    embedder.load_model()

    embedding_dim = embedder.get_dimension()
    console.print(f"   Embedding dimension: {embedding_dim}")

    texts = [chunk["content"] for chunk in all_chunks]
    embeddings = embedder.embed_batch(texts, show_progress=True)

    console.print()

    if remote_only:
        console.print("⚡ Remote-only mode: skipping local database", style="bold blue")
        console.print()

        local_data = prepare_data_structure(all_chunks, embeddings)
        db_manager = None
    else:
        output_path = config.get_output_path().absolute()
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"💾 Creating ChromaDB at: {output_path}")

        db_manager = LocalDBManager(str(output_path), config.get_db_name(), config.verbose)

        if not db_manager.create_database():
            console.print("❌ Failed to create database", style="red")
            sys.exit(1)

        chunks_added = db_manager.add_chunks(all_chunks, embeddings)

        if chunks_added > 0:
            stats = db_manager.get_stats()
            console.print()
            console.print("✅ ChromaDB created successfully!", style="bold green")
            console.print(f"   📊 Total chunks: {stats.get('total_chunks', chunks_added)}")
            console.print(f"   📁 Location: {output_path}")
            console.print()
        else:
            console.print("❌ Failed to add chunks to database", style="red")
            sys.exit(1)

        local_data = db_manager.get_all_chunks()

    if local_data and "ids" in local_data:
        console.print("🔄 Syncing to remote ChromaDB...")

        if remote_only:
            console.print(f"📊 Syncing {len(local_data.get('ids', []))} chunks to remote...", style="cyan")

        sync_manager = RemoteSyncManager(
            host=config.remote_host,
            port=config.remote_port,
            tenant=config.remote_tenant,
            database=config.remote_database,
            api_key=config.api_key,
            verbose=config.verbose,
        )

        if reset_all_remote:
            console.print()
            console.print(f"🗑️  Resetting ALL collections on remote server...", style="yellow")
            deleted_count = sync_manager.delete_all_collections()
            console.print()

        synced = sync_manager.sync_collection(
            collection_name=config.get_db_name(),
            local_data=local_data,
            metadata={
                "created_at": None,
                "description": f"ChromaDB collection for {config.get_db_name()}",
            },
            reset_remote=reset_remote,
        )

        if synced:
            console.print()
            console.print("✅ Sync completed successfully!", style="bold green")
        else:
            console.print()
            if remote_only:
                console.print("⚠️  Sync was skipped or failed.", style="yellow")
            else:
                console.print("⚠️  Sync was skipped or failed. Local database is ready.", style="yellow")

        if db_manager:
            db_manager.close()


if __name__ == "__main__":
    app()