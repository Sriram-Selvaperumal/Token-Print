"""Unit tests for Pydantic API schemas and response structures."""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend directory to sys.path so tests can be run directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from pydantic import ValidationError

from app.schemas import (
    AblateRequest,
    AnalyzeImageRequest,
    AnalyzeRequest,
    AnalyzeResponse,
    HFInspectResponse,
    HFModelMeta,
    HFSearchResponse,
    ModelInfo,
    PatchRequest,
    Projection,
    ProvenanceInfo,
    RagAnalyzeRequest,
    RagAnalyzeResponse,
    RagChunk,
    Token,
)


def test_analyze_request_schema():
    req = AnalyzeRequest(sentence="Hello world")
    assert req.sentence == "Hello world"

    with pytest.raises(ValidationError):
        AnalyzeRequest(sentence="")


def test_analyze_image_request_schema():
    req = AnalyzeImageRequest(image="data:image/png;base64,iVBORw0KGgo=")
    assert req.image.startswith("data:image/")

    with pytest.raises(ValidationError):
        AnalyzeImageRequest(image="")


def test_ablate_request_schema():
    req = AblateRequest(sentence="Test prompt", zero_layers=[0, 1], zero_heads={"0": [1, 2]})
    assert req.sentence == "Test prompt"
    assert req.zero_layers == [0, 1]
    assert req.zero_heads == {"0": [1, 2]}


def test_patch_request_schema():
    req = PatchRequest(sentence="Target", source_sentence="Source", patch_layers=[1, 2])
    assert req.sentence == "Target"
    assert req.source_sentence == "Source"
    assert req.patch_layers == [1, 2]

    with pytest.raises(ValidationError):
        PatchRequest(sentence="Target", source_sentence="Source", patch_layers=[])


def test_token_and_provenance_schema():
    tok = Token(index=0, text="Hello", piece="Hello", id=15496, is_special=False)
    assert tok.index == 0
    assert tok.text == "Hello"
    assert not tok.is_special

    prov = ProvenanceInfo(
        source_type="REAL",
        backend="hf_local",
        device="cpu",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
    )
    assert prov.source_type == "REAL"
    assert prov.backend == "hf_local"

    with pytest.raises(ValidationError):
        ProvenanceInfo(source_type="INVALID_TYPE", backend="hf_local", device="cpu")


def test_analyze_response_schema():
    tok = Token(index=0, text="Test", piece="Test", id=1, is_special=False)
    resp = AnalyzeResponse(
        sentence="Test",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        num_layers=1,
        num_heads=1,
        hidden_size=64,
        tokens=[tok],
        attention=[[[[1.0]]]],
    )
    assert resp.sentence == "Test"
    assert resp.num_layers == 1
    assert len(resp.tokens) == 1
    assert resp.attention[0][0][0][0] == 1.0


def test_rag_schemas():
    chunk = RagChunk(id="c1", text="Chunk context text")
    assert chunk.id == "c1"

    req = RagAnalyzeRequest(
        query="What is TokenPrint?",
        chunks=[chunk],
        reduction_mode="both",
        ungrounded_threshold=0.15,
    )
    assert req.query == "What is TokenPrint?"
    assert len(req.chunks) == 1
    assert req.ungrounded_threshold == 0.15

    # Test threshold validation
    with pytest.raises(ValidationError):
        RagAnalyzeRequest(query="Q", chunks=[chunk], ungrounded_threshold=1.5)

    tok = Token(index=0, text="Q", piece="Q", id=10, is_special=False)
    rag_resp = RagAnalyzeResponse(
        sentence="Combined query and chunks",
        query="What is TokenPrint?",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        num_layers=1,
        num_heads=1,
        hidden_size=64,
        tokens=[tok],
        attention=[[[[1.0]]]],
        chunk_spans={"c1": [0, 1]},
        ungrounded=[False],
    )
    assert rag_resp.query == "What is TokenPrint?"
    assert rag_resp.chunk_spans["c1"] == [0, 1]


def test_hf_schemas():
    meta = HFModelMeta(id="owner/test-model", downloads=100, likes=10)
    assert meta.id == "owner/test-model"
    assert meta.downloads == 100

    search = HFSearchResponse(query="qwen", limit=1, models=[meta])
    assert search.query == "qwen"
    assert len(search.models) == 1

    inspect = HFInspectResponse(
        model_id="owner/test-model",
        architecture="Qwen2ForCausalLM",
        compatibility_level="High",
    )
    assert inspect.architecture == "Qwen2ForCausalLM"
    assert inspect.compatibility_level == "High"


def test_model_info_schema():
    info = ModelInfo(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        num_layers=24,
        num_heads=14,
        hidden_size=896,
        attn_implementation="sdpa",
        max_tokens=2048,
        ready=True,
    )
    assert info.ready is True
    assert info.hidden_size == 896

