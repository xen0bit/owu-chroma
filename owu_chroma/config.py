from pathlib import Path
from typing import Optional


class Config:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        output_path: Optional[Path] = None,
        db_name: Optional[str] = None,
        verbose: bool = False,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8080,
        remote_tenant: Optional[str] = None,
        remote_database: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.output_path = output_path
        self.db_name = db_name
        self.verbose = verbose
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_tenant = remote_tenant
        self.remote_database = remote_database
        self.api_key = api_key

    def get_output_path(self) -> Path:
        if self.output_path:
            db_dir = self.output_path
        else:
            db_dir = Path.cwd()
        return db_dir / f"{self.get_db_name()}.chroma"

    def get_db_name(self) -> str:
        if self.db_name:
            return self.db_name
        if self.output_path:
            return self.output_path.stem
        return "unnamed"

    def get_remote_url(self) -> str:
        return f"http://{self.remote_host}:{self.remote_port}"