"""Core contracts for the paper-based C-DIC reproduction."""

from cdic_repro.config import RetrievalConfig, SupportOrder
from cdic_repro.credit import (
    CompressionGradientPlan,
    CreditPlan,
    build_compression_gradient_plan,
    build_credit_plan,
)
from cdic_repro.engine import CdicInferenceEngine, TurnOutput
from cdic_repro.icae_adapter import (
    IcaeV1AdapterConfig,
    IcaeV1InferenceAdapter,
    IcaeV1TrainingAdapter,
)
from cdic_repro.memory_state import MemoryBank, ThreadState
from cdic_repro.model_protocol import (
    CdicModelAdapter,
    CdicTrainingAdapter,
    CompressedTurn,
    TrainingLoss,
)
from cdic_repro.msc import MscEpisode, MscTurn, load_msc_episodes
from cdic_repro.retrieval import RetrievalResult, ScoreRecord, retrieve
from cdic_repro.training import CdicTrainingEngine, EpisodeTrainingResult
from cdic_repro.writeback import WriteAction, WriteBackResult, apply_write_back

__all__ = [
    "CdicInferenceEngine",
    "CdicModelAdapter",
    "CdicTrainingAdapter",
    "CdicTrainingEngine",
    "CompressedTurn",
    "CompressionGradientPlan",
    "CreditPlan",
    "IcaeV1AdapterConfig",
    "IcaeV1InferenceAdapter",
    "IcaeV1TrainingAdapter",
    "MemoryBank",
    "MscEpisode",
    "MscTurn",
    "EpisodeTrainingResult",
    "RetrievalConfig",
    "RetrievalResult",
    "ScoreRecord",
    "SupportOrder",
    "ThreadState",
    "TrainingLoss",
    "TurnOutput",
    "WriteAction",
    "WriteBackResult",
    "apply_write_back",
    "build_compression_gradient_plan",
    "build_credit_plan",
    "load_msc_episodes",
    "retrieve",
]
