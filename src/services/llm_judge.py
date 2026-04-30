import json
from typing import Dict, Any
from src.schemas.models import SCIPABOutput, CorrectiveVector
from src.config.settings import settings

SCIPAB_SYSTEM_PROMPT = """
You are a Senior Quantitative Analyst and Trading Judge. Your task is to evaluate a trading signal and provide a corrective adjustment vector using the SCIPAB framework.

SCIPAB Framework:
- Situation: The current market state (Technicals, Fundamentals, Sentiment).
- Complication: Why the current base signal might be risky or suboptimal.
- Implication: What happens if the error in the base signal is ignored.
- Position: Your authoritative stance on the required adjustment.
- Action: Specific corrective steps needed.
- Benefit: The expected improvement in risk-adjusted returns (Sharpe/Sortino).

Input Data:
{market_context}
Base Signal: {base_signal_info}

Requirement:
Output MUST be a valid JSON object matching the following structure:
{{
  "situation": "...",
  "complication": "...",
  "implication": "...",
  "position": "...",
  "action": "...",
  "benefit": "...",
  "reasoning": "Detailed technical justification",
  "corrective_vector": {{
    "buy_prob_delta": float (-1.0 to 1.0),
    "sell_prob_delta": float (-1.0 to 1.0),
    "position_size_delta": float (-1.0 to 1.0),
    "confidence_delta": float (-1.0 to 1.0),
    "fundamental_alignment": float (0.0 to 1.0),
    "technical_consistency": float (0.0 to 1.0),
    "sentiment_shock": float (0.0 to 1.0),
    "macro_risk": float (0.0 to 1.0),
    "liquidity_risk": float (0.0 to 1.0),
    "tail_risk": float (0.0 to 1.0),
    "overall_sanity_score": float (0.0 to 1.0),
    "extra_features": [53 floats representing specialized embedding deltas]
  }}
}}
Ensure the output is strictly valid JSON and adheres to the SCIPAB logic.
"""

class LLMJudgeService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.LLM_API_KEY
        # In a real implementation, you'd use 'instructor' or 'guidance' here.
        # We'll mock the LLM call for the skeleton.
    
    async def judge_signal(self, market_data: Dict, base_signal: Dict) -> SCIPABOutput:
        prompt = SCIPAB_SYSTEM_PROMPT.format(
            market_context=json.dumps(market_data),
            base_signal_info=json.dumps(base_signal)
        )
        # Mock LLM API call
        # response = await openai_client.chat.completions.create(..., messages=[{"role": "system", "content": prompt}])
        # return SCIPABOutput.model_validate_json(response.choices[0].message.content)
        
        # Return a dummy SCIPAB output for structure reference
        return SCIPABOutput(
            situation="Market is in a high-volatility sideways regime.",
            complication="Base signal ignores the upcoming CPI release and macro tail risk.",
            implication="High probability of stop-loss hunt in the next 4 hours.",
            position="Reduce exposure and wait for confirmation.",
            action="Decrease position size delta by 0.3.",
            benefit="Lowered drawdown and better entry price.",
            reasoning="Technical indicators show overbought RSI combined with macro headwinds.",
            corrective_vector=CorrectiveVector(
                buy_prob_delta=-0.1,
                sell_prob_delta=0.05,
                position_size_delta=-0.3,
                confidence_delta=-0.2,
                fundamental_alignment=0.4,
                technical_consistency=0.8,
                sentiment_shock=0.1,
                macro_risk=0.7,
                liquidity_risk=0.2,
                tail_risk=0.6,
                overall_sanity_score=0.5,
                extra_features=[0.0] * 53
            )
        )
