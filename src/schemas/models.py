from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime

class CorrectiveVector(BaseModel):
    # Signal deltas (4 dimensions)
    buy_prob_delta: float = Field(..., description="Change in buy probability")
    sell_prob_delta: float = Field(..., description="Change in sell probability")
    position_size_delta: float = Field(..., description="Change in recommended position size")
    confidence_delta: float = Field(..., description="Change in confidence score")
    
    # Sanity & Risk Scores (7 dimensions)
    fundamental_alignment: float = Field(..., ge=0, le=1)
    technical_consistency: float = Field(..., ge=0, le=1)
    sentiment_shock: float = Field(..., ge=0, le=1)
    macro_risk: float = Field(..., ge=0, le=1)
    liquidity_risk: float = Field(..., ge=0, le=1)
    tail_risk: float = Field(..., ge=0, le=1)
    overall_sanity_score: float = Field(..., ge=0, le=1)
    
    # Embedding slice and feature importance deltas (53 dimensions)
    # To satisfy the 64-dimensional requirement, we use a fixed-size list for the remainder
    extra_features: List[float] = Field(..., min_length=53, max_length=53)

    def to_tensor(self):
        import torch
        base = [
            self.buy_prob_delta, self.sell_prob_delta, self.position_size_delta, self.confidence_delta,
            self.fundamental_alignment, self.technical_consistency, self.sentiment_shock,
            self.macro_risk, self.liquidity_risk, self.tail_risk, self.overall_sanity_score
        ]
        return torch.tensor(base + self.extra_features)

class SCIPABOutput(BaseModel):
    situation: str
    complication: str
    implication: str
    position: str
    action: str
    benefit: str
    corrective_vector: CorrectiveVector
    reasoning: str

class MemPalaceEntry(BaseModel):
    id: str
    timestamp: datetime
    symbol: str
    timeframe: str
    signal_data: Dict
    corrective_vector: List[float]
    feedback: Optional[str] = None
    embedding: List[float]

class SignalResponse(BaseModel):
    symbol: str
    timestamp: datetime
    base_signal: str
    base_confidence: float
    critic_correction: Optional[CorrectiveVector]
    final_decision: str
