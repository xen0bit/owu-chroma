# owu-chroma

Create and sync ChromaDB vector databases from ZIP files.

## Features

- Extract and process ZIP files containing text documents
- Automatic chunking of content based on file type
- Create embeddings using sentence-transformers models
- Store vector databases locally with ChromaDB
- Sync to remote ChromaDB servers
- Support for custom chunk sizes, overlap, and embedding models

## Installation

This project is managed with uv. Run it directly with uvx:

```bash
uvx owu-chroma myproject.zip
```

To install locally:

```bash
uv sync
uv run owu-chroma myproject.zip
```

## Usage

### Basic Usage

Process a ZIP file and create a vector database:

```bash
uvx owu-chroma myproject.zip
```

### Advanced Options

```bash
uvx owu-chroma myproject.zip \
  --name mydb \
  --chunk-size 1000 \
  --chunk-overlap 100 \
  --model all-MiniLM-L6-v2 \
  --output-dir ./output \
  --remote-host 192.168.1.100 \
  --remote-port 8080 \
  --verbose
```

## Commands

### `uvx owu-chroma <zip-file>`

Main command to process ZIP files and create/sync vector databases.

**Arguments:**
- `zip-file` - Path to the ZIP file to process

**Options:**
- `--name, -n` - Database name (default: derived from zip filename)
- `--chunk-size, -s` - Chunk size in characters (default: 1000)
- `--chunk-overlap, -o` - Chunk overlap in characters (default: 100)
- `--model, -m` - Embedding model name (default: all-MiniLM-L6-v2)
- `--output-dir` - Output directory path (default: current directory)
- `--remote-host` - Remote ChromaDB host (default: 127.0.0.1)
- `--remote-port` - Remote ChromaDB port (default: 8080)
- `--remote-tenant` - Remote ChromaDB tenant (optional)
- `--remote-database` - Remote ChromaDB database (optional)
- `--api-key` - API key for remote ChromaDB (optional)
- `--verbose, -v` - Show detailed progress
- `--reset-remote, -r` - Reset remote collection (delete and recreate)
- `--reset-all, -R` - Delete ALL collections on remote server before syncing
- `--cpu, -c` - Force CPU usage for embeddings (default: auto-detect)

## Dependencies

- uv (Python package manager)
- chromadb>=0.5.0
- sentence-transformers>=3.0.0
- typer>=0.12.0
- rich>=13.0.0

## License

MIT