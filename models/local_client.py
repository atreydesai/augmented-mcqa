"""Local model client using vLLM for Augmented MCQA."""

from typing import Optional, List, Any
import os
import time

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
        max_model_len: Optional[int] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        enable_chunked_prefill: Optional[bool] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[str | list[str]] = None,
        stop_token_ids: Optional[list[int]] = None,
    ):
        """
        Initialize local vLLM client.
        
        Args:
            model_id: HuggingFace model ID or local path
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: all available)
            seed: Random seed for reproducibility
            dtype: Data type ("auto", "float16", "bfloat16")
            max_model_len: Optional maximum sequence length override
            tokenizer_mode: vLLM tokenizer mode ("auto" or "slow")
            trust_remote_code: Whether to trust model-provided custom code
            max_num_batched_tokens: vLLM scheduler token budget cap
            max_num_seqs: vLLM scheduler sequence cap
            enable_chunked_prefill: Enable/disable chunked prefill in vLLM
            temperature: Default generation temperature
            top_p: Default top-p sampling value
            top_k: Default top-k sampling value
            min_p: Default min-p sampling value
            repetition_penalty: Default repetition penalty
            presence_penalty: Default presence penalty
            frequency_penalty: Default frequency penalty
            stop: Default stop string(s)
            stop_token_ids: Default stop token IDs
        """
        self._model_id = model_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._seed = seed
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._tokenizer_mode = tokenizer_mode
        self._trust_remote_code = trust_remote_code
        self._max_num_batched_tokens = (
            max_num_batched_tokens
            if max_num_batched_tokens is not None
            else self._env_int("VLLM_MAX_NUM_BATCHED_TOKENS")
        )
        self._max_num_seqs = (
            max_num_seqs if max_num_seqs is not None else self._env_int("VLLM_MAX_NUM_SEQS")
        )
        self._enable_chunked_prefill = (
            enable_chunked_prefill
            if enable_chunked_prefill is not None
            else self._env_bool("VLLM_ENABLE_CHUNKED_PREFILL")
        )
        self._sampling_defaults: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop": stop,
            "stop_token_ids": stop_token_ids,
        }
        
        # Lazy init - don't load model until first generation
        self._llm = None
        self._tokenizer = None
        self._tensor_parallel_size = tensor_parallel_size
        self._init_error: Optional[Exception] = None
        self._model_load_seconds: float = 0.0
        
        # Set HF cache directory
        os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR / "transformers")

    @staticmethod
    def _env_int(name: str) -> Optional[int]:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return None
        return int(raw)

    @staticmethod
    def _env_bool(name: str) -> Optional[bool]:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return None
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for {name}: {raw}")
    
    def _ensure_loaded(self):
        """Load the model if not already loaded."""
        if self._llm is not None:
            return
        if self._init_error is not None:
            raise RuntimeError(
                f"Previous local model initialization failed for {self._model_id}"
            ) from self._init_error

        self._ensure_transformers_tokenizer_compat()
        import torch
        from vllm import LLM
        
        # Determine tensor parallel size
        if self._tensor_parallel_size is None:
            self._tensor_parallel_size = max(1, torch.cuda.device_count())
        
        print(f"Loading model: {self._model_id}")
        print(f"  GPU memory utilization: {self._gpu_memory_utilization}")
        print(f"  Tensor parallel size: {self._tensor_parallel_size}")
        print(f"  Seed: {self._seed}")
        print(f"  Dtype: {self._dtype}")
        print(f"  Tokenizer mode: {self._tokenizer_mode}")
        if self._max_model_len is not None:
            print(f"  Max model len: {self._max_model_len}")
        if self._max_num_batched_tokens is not None:
            print(f"  Max num batched tokens: {self._max_num_batched_tokens}")
        if self._max_num_seqs is not None:
            print(f"  Max num seqs: {self._max_num_seqs}")
        if self._enable_chunked_prefill is not None:
            print(f"  Chunked prefill: {self._enable_chunked_prefill}")
        
        # vLLM automatically uses Flash Attention when available
        llm_kwargs = dict(
            model=self._model_id,
            gpu_memory_utilization=self._gpu_memory_utilization,
            tensor_parallel_size=self._tensor_parallel_size,
            seed=self._seed,
            dtype=self._dtype,
            download_dir=str(MODEL_CACHE_DIR),
            trust_remote_code=self._trust_remote_code,
            tokenizer_mode=self._tokenizer_mode,
        )
        if self._max_model_len is not None:
            llm_kwargs["max_model_len"] = self._max_model_len
        if self._max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = self._max_num_batched_tokens
        if self._max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = self._max_num_seqs
        if self._enable_chunked_prefill is not None:
            llm_kwargs["enable_chunked_prefill"] = self._enable_chunked_prefill
        load_start = time.perf_counter()
        try:
            self._llm = LLM(**llm_kwargs)
        except Exception as exc:
            # Some local-model tokenizers fail under slow mode in vLLM; retry once with auto.
            retry_with_auto = (
                self._tokenizer_mode != "auto"
                and "all_special_tokens_extended" in str(exc)
            )
            if not retry_with_auto:
                self._init_error = exc
                raise

            print(
                "  Tokenizer initialization failed with current mode; "
                "retrying once with tokenizer_mode=auto"
            )
            llm_kwargs["tokenizer_mode"] = "auto"
            self._tokenizer_mode = "auto"
            try:
                self._llm = LLM(**llm_kwargs)
            except Exception as retry_exc:
                self._init_error = retry_exc
                raise

        self._model_load_seconds = max(0.0, time.perf_counter() - load_start)
        print(f"  Model loaded successfully in {self._model_load_seconds:.2f}s")

    @staticmethod
    def _ensure_transformers_tokenizer_compat() -> None:
        """Patch TokenizersBackend for vLLM compatibility on newer Transformers."""
        try:
            from transformers.tokenization_utils_tokenizers import TokenizersBackend
        except Exception:
            return

        patched = False

        if not hasattr(TokenizersBackend, "all_special_tokens_extended"):
            @property
            def all_special_tokens_extended(self):  # type: ignore[override]
                return list(self.all_special_tokens)

            TokenizersBackend.all_special_tokens_extended = all_special_tokens_extended  # type: ignore[attr-defined]
            patched = True

        if not hasattr(TokenizersBackend, "special_tokens_map_extended"):
            @property
            def special_tokens_map_extended(self):  # type: ignore[override]
                return dict(self.special_tokens_map)

            TokenizersBackend.special_tokens_map_extended = special_tokens_map_extended  # type: ignore[attr-defined]
            patched = True

        if patched:
            print("  Applied TokenizersBackend compatibility patch for vLLM")
    
    @property
    def name(self) -> str:
        return f"Local ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_load_seconds(self) -> float:
        return self._model_load_seconds
    
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
        
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "seed": kwargs.get("seed", self._seed),
        }
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
            "stop_token_ids",
            "ignore_eos",
        ):
            value = kwargs.get(key, self._sampling_defaults.get(key))
            if value is not None:
                sampling_kwargs[key] = value

        repeat_penalty = kwargs.get("repeat_penalty")
        if repeat_penalty is not None and "repetition_penalty" not in sampling_kwargs:
            sampling_kwargs["repetition_penalty"] = repeat_penalty

        sampling_params = SamplingParams(**sampling_kwargs)

        # Use chat() when available so instruct models receive their chat template.
        # Without it, OLMo/Qwen etc. hit EOS immediately and return empty/single-token outputs.
        if hasattr(self._llm, "chat"):
            messages = [[{"role": "user", "content": p}] for p in prompts]
            outputs = self._llm.chat(messages, sampling_params)
        else:
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
        self._init_error = None
