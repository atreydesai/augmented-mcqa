Archived faulty evaluation datasets.

Reason:
The GPT-5.4-mini medium/xhigh, together/openai/gpt-oss-120b, and vllm/Qwen/Qwen3-14B evaluation runs were found to be faulty because responses hit the max token limit.

Important details:
- The CLI default for --max-tokens is 2048.
- These folders were moved out of active collected dataset roots so analyze-irt discovery will not include them.
- Re-run replacements should use a larger --max-tokens value. For Qwen3-14B, use --max-tokens 32768.

Archived from:
- datasets/collected/.../openai_gpt-5.4-mini
- datasets/collected/.../together_openai_gpt-oss-120b
- datasets/collected/.../vllm_Qwen_Qwen3-14B
- datasets/collected_gptmini_xhigh/.../openai_gpt-5.4-mini
