"""Core contracts for the paper-based C-DIC reproduction."""

from cdic_repro.config import RetrievalConfig, SupportOrder
from cdic_repro.credit import CreditPlan, build_credit_plan
from cdic_repro.engine import CdicInferenceEngine, TurnOutput
from cdic_repro.icae_adapter import IcaeV1AdapterConfig, IcaeV1InferenceAdapter
from cdic_repro.memory_state import MemoryBank, ThreadState
from cdic_repro.model_protocol import CdicModelAdapter, CompressedTurn
from cdic_repro.retrieval import RetrievalResult, ScoreRecord, retrieve
from cdic_repro.writeback import WriteAction, WriteBackResult, apply_write_back

__all__ = [
    "CdicInferenceEngine",
    "CdicModelAdapter",
    "CompressedTurn",
    "CreditPlan",
    "IcaeV1AdapterConfig",
    "IcaeV1InferenceAdapter",
    "MemoryBank",
    "RetrievalConfig",
    "RetrievalResult",
    "ScoreRecord",
    "SupportOrder",
    "ThreadState",
    "TurnOutput",
    "WriteAction",
    "WriteBackResult",
    "apply_write_back",
    "build_credit_plan",
    "retrieve",
]
