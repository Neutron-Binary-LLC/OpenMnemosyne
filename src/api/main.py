from fastapi import FastAPI, Depends, HTTPException
from src.schemas.models import SignalResponse, CorrectiveVector, SCIPABOutput
from src.services.llm_judge import LLMJudgeService
from src.models.neural_critic import get_neural_critic_model
from src.models.base_signal import BaseSignalGenerator
from typing import Dict
import torch
from datetime import datetime

app = FastAPI(title="OpenMnemosyne Inference Gateway")

# Dependency injection for models and services
critic_model = get_neural_critic_model()
base_model = BaseSignalGenerator()
llm_service = LLMJudgeService()

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/generate_signal", response_model=SignalResponse)
async def generate_signal(symbol: str, timeframe: str, data: Dict):
    # 1. Prepare data tensor (simplified)
    input_tensor = torch.randn(1, 16, 128) # Mock market data
    
    # 2. Get Base Signal
    with torch.no_grad():
        base_probs = base_model(torch.randn(1, 16, 64))
        base_signal_idx = torch.argmax(base_probs).item()
        base_signal = ["SELL", "HOLD", "BUY"][base_signal_idx]
        base_conf = base_probs[0, base_signal_idx].item()

    # 3. Get Critic Correction
    with torch.no_grad():
        correction_vec = critic_model(input_tensor)
        # Convert 64D tensor to CorrectiveVector pydantic model
        # (Simplified conversion for the skeleton)
        correction = CorrectiveVector(
            buy_prob_delta=correction_vec[0,0].item(),
            sell_prob_delta=correction_vec[0,1].item(),
            position_size_delta=correction_vec[0,2].item(),
            confidence_delta=correction_vec[0,3].item(),
            fundamental_alignment=torch.sigmoid(correction_vec[0,4]).item(),
            technical_consistency=torch.sigmoid(correction_vec[0,5]).item(),
            sentiment_shock=torch.sigmoid(correction_vec[0,6]).item(),
            macro_risk=torch.sigmoid(correction_vec[0,7]).item(),
            liquidity_risk=torch.sigmoid(correction_vec[0,8]).item(),
            tail_risk=torch.sigmoid(correction_vec[0,9]).item(),
            overall_sanity_score=torch.sigmoid(correction_vec[0,10]).item(),
            extra_features=correction_vec[0, 11:].tolist()[:53]
        )

    return SignalResponse(
        symbol=symbol,
        timestamp=datetime.now(),
        base_signal=base_signal,
        base_confidence=base_conf,
        critic_correction=correction,
        final_decision=base_signal # Logical fusion would happen here
    )

@app.post("/get_critic_correction", response_model=CorrectiveVector)
async def get_critic_correction(market_data: Dict):
    # Dedicated endpoint for pure critic output
    input_tensor = torch.randn(1, 16, 128)
    with torch.no_grad():
        vec = critic_model(input_tensor)
        return CorrectiveVector(
            buy_prob_delta=vec[0,0].item(),
            sell_prob_delta=vec[0,1].item(),
            position_size_delta=vec[0,2].item(),
            confidence_delta=vec[0,3].item(),
            fundamental_alignment=0.9, technical_consistency=0.9, sentiment_shock=0.1,
            macro_risk=0.1, liquidity_risk=0.1, tail_risk=0.1, overall_sanity_score=0.9,
            extra_features=[0.0]*53
        )
