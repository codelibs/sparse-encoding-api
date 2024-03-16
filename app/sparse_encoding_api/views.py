import time
from typing import List, Dict

from fastapi import APIRouter, Depends

import sparse_encoding_api
from sparse_encoding_api.models import Docs

from .services import SparseEncodingService, get_sparse_encoding_service

api_router = APIRouter()


@api_router.get("/")
async def index(sparse_encoding_service: SparseEncodingService = Depends(get_sparse_encoding_service)):
    return {
        "version": sparse_encoding_api.__version__,
        "model_name": sparse_encoding_service.get_model_name(),
    }


@api_router.get("/ping")
async def ping():
    return {
        "status": "ok"
    }


@api_router.post("/encode")
async def encode(
    docs: Docs, sparse_encoding_service: SparseEncodingService = Depends(get_sparse_encoding_service)
):
    start: float = time.time()
    embeddings: List[Dict[str, float]] = sparse_encoding_service.encode(docs)
    return {
        "took": time.time() - start,
        "embeddings": embeddings,
    }
