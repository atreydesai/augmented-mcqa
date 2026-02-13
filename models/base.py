"""
Base model client interface for Augmented MCQA.

Defines the abstract interface that all model clients must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class GenerationResult:
    """Result from a model generation call."""
    text: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    
    def extract_answer(self) -> str:
        """
        Extract the answer letter from the response text.
        
        Looks for patterns like "The answer is (A)" or "The answer is A".
        """
        import re
        
        text = self.text.strip()
        
        # Pattern 1: "The answer is (X)" or "The answer is X"
        pattern1 = r"[Tt]he answer is \(?([A-J])\)?"
        match = re.search(pattern1, text)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: Last single capital letter in the response
        pattern2 = r"\b([A-J])\b"
        matches = re.findall(pattern2, text)
        if matches:
            return matches[-1].upper()
        
        # Pattern 3: Just return empty if no match
        return ""


class ModelClient(ABC):
    """
    Abstract base class for model clients.
    
    All model providers (OpenAI, Anthropic, Gemini, local) implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this client."""
        pass
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """The model identifier used for API calls."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            GenerationResult with the generated text
        """
        pass
    
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 100,
        **kwargs,
    ) -> list[GenerationResult]:
        """
        Generate responses for a batch of prompts.
        
        Default implementation calls generate() sequentially.
        Subclasses may override for more efficient batch processing.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per response
            **kwargs: Additional parameters
            
        Returns:
            List of GenerationResults
        """
        return [
            self.generate(prompt, max_tokens, **kwargs)
            for prompt in prompts
        ]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id})"
