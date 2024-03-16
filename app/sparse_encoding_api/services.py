import logging
import os
from typing import List, Dict

from sparse_encoding_api.models import Docs
from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch

logger = logging.getLogger(__name__)


class NeuralSparseModel(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = self.bert(
            input_ids=input["input_ids"], attention_mask=input["attention_mask"])[0]
        values, _ = torch.max(
            result*input["attention_mask"].unsqueeze(-1), dim=1)
        values = torch.log(1 + torch.relu(values))
        return {"output": values}


class SparseEncodingService:
    def __init__(self) -> None:
        self._model_name: str = os.getenv(
            "MODEL_NAME", "naver/splade_v2_distil")
        logger.info(f"Loading {self._model_name}")
        cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/code/model")
        device: (str | None) = os.getenv("DEVICE")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, cache_dir=cache_dir, device=device)
        bert = AutoModelForMaskedLM.from_pretrained(
            self._model_name, cache_dir=cache_dir)
        if device is not None:
            self._tokenizer.to(device)
            bert.to(device)
        self._model = NeuralSparseModel(bert)

    def get_model_name(self) -> str:
        return self._model_name

    def encode(self, docs: Docs) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for sentence in docs.sentences:
            tokens = self._tokenizer(
                sentence,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt")
            vec = self._model(tokens)["output"][0]
            cols = vec.nonzero().squeeze().cpu().tolist()
            weights = vec[cols].cpu().tolist()
            idx2token = {idx: token for token,
                         idx in self._tokenizer.get_vocab().items()}
            sparse_dict_tokens = {idx2token[idx]: round(
                weight, 2) for idx, weight in zip(cols, weights)}
            sparse_dict_tokens = {k: v for k, v in sorted(
                sparse_dict_tokens.items(), key=lambda item: item[1], reverse=True)}
            results.append(sparse_dict_tokens)

        return results


sparse_encoding_service: SparseEncodingService = SparseEncodingService()


async def get_sparse_encoding_service() -> SparseEncodingService:
    return sparse_encoding_service
