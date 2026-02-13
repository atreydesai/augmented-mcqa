"""
Local model client using vLLM for Augmented MCQA.

Flash Attention is enabled by default in vLLM.
"""

from typing import Optional, List
import os

from .base import ModelClient, GenerationResult
from config import MODEL_CACHE_DIR, RANDOM_SEED


class LocalClient(ModelClient):
    """
    Local model client using vLLM for inference.
    
    Flash Attention is automatically enabled by vLLM when available.
    Uses PagedAttention for efficient KV cache management.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: Optional[int] = None,
        seed: int = RANDOM_SEED,
        dtype: str = "auto",
    ):
        """
        Initialize local vLLM client.
        
        Args:
            model_id: HuggingFace model ID or local path
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: all available)
            seed: Random seed for reproducibility
            dtype: Data type ("auto", "float16", "bfloat16")
        """
        self._model_id = model_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._seed = seed
        self._dtype = dtype
        
        # Lazy init - don't load model until first generation
        self._llm = None
        self._tokenizer = None
        self._tensor_parallel_size = tensor_parallel_size
        
        # Set HF cache directory
        os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR / "transformers")
    
    def _ensure_loaded(self):
        """Load the model if not already loaded."""
        if self._llm is not None:
            return
        
        import torch
        from vllm import LLM
        
        # Determine tensor parallel size
        if self._tensor_parallel_size is None:
            self._tensor_parallel_size = torch.cuda.device_count()
        
        print(f"Loading model: {self._model_id}")
        print(f"  GPU memory utilization: {self._gpu_memory_utilization}")
        print(f"  Tensor parallel size: {self._tensor_parallel_size}")
        print(f"  Seed: {self._seed}")
        
        # vLLM automatically uses Flash Attention when available
        # No need to explicitly enable it
        self._llm = LLM(
            model=self._model_id,
            gpu_memory_utilization=self._gpu_memory_utilization,
            tensor_parallel_size=self._tensor_parallel_size,
            seed=self._seed,
            dtype=self._dtype,
            download_dir=str(MODEL_CACHE_DIR),
            trust_remote_code=True,
        )
        
        print(f"  Model loaded successfully")
    
    @property
    def name(self) -> str:
        return f"Local ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using vLLM.
        """
        results = self.generate_batch([prompt], max_tokens, **kwargs)
        return results[0]
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Efficient batch generation using vLLM.
        
        This is much more efficient than calling generate() in a loop
        as it processes all prompts in a single forward pass.
        """
        from vllm import SamplingParams
        
        self._ensure_loaded()
        
        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            seed=self._seed,
        )
        
        # Generate
        outputs = self._llm.generate(prompts, sampling_params)
        
        # Convert to GenerationResults
        results = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            finish_reason = output.outputs[0].finish_reason if output.outputs else None
            
            results.append(GenerationResult(
                text=text,
                model=self._model_id,
                finish_reason=finish_reason,
                usage=None,  # vLLM doesn't provide token counts in the same way
                raw_response=output,
            ))
        
        return results
    
    def unload(self):
        """Unload the model to free GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            
            import torch
            torch.cuda.empty_cache()
            print(f"Model unloaded: {self._model_id}")
