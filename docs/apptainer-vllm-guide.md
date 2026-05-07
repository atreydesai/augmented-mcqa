# Running vLLM with Apptainer for Inspect Evaluations

This guide explains how to run recent open-weight models on Nexus/CLIP using
vLLM inside Apptainer, while keeping Inspect AI and this repo's `.venv` on the
host. It assumes you are working from:

```bash
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa
```

The short version:

1. Download model weights into the shared Hugging Face cache.
2. Start one vLLM Apptainer server per model, usually on an RTX A6000 GPU.
3. Submit Inspect slices as ordinary API-client jobs using `--model-base-url`.
4. Let the generated finalizer collect the `.eval` logs into the normal
   evaluated dataset outputs.

Apptainer is only used for the vLLM server. The evaluation slices do not run
inside Apptainer.

## Why Apptainer Is Needed

Nexus runs RHEL 8.10 with glibc 2.28. Recent vLLM, PyTorch, CUDA extension, and
transformers wheels are built against newer glibc versions. Installing current
vLLM directly into the repo `.venv` can fail at install time or import time with
errors like:

```text
version `GLIBC_2.32' not found
```

Apptainer avoids that problem by running the vLLM Docker image as a
cluster-friendly container. The container brings its own Python, glibc, PyTorch,
CUDA userspace libraries, transformers, and vLLM. The host still provides the
Linux kernel, mounted filesystems, and NVIDIA driver through `--nv`.

The architecture is:

```text
GPU node
  Apptainer container
    vLLM OpenAI-compatible HTTP server
    model weights mounted from /fs/clip-scratch/adesai10/hub
    listens on http://<gpu-node>:<port>/v1

Login/CPU/ordinary compute nodes
  repo .venv
  Inspect AI
  submit-evaluate-cluster slices
  HTTP requests to the vLLM server
```

Because Inspect talks to vLLM over HTTP, the host `.venv` does not need to load
vLLM, PyTorch, or CUDA extensions.

## Important Paths

Use these paths unless you have a specific reason to change them:

```bash
PROJECT_DIR=/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa
HF_ROOT=/fs/clip-scratch/adesai10
HF_HUB_CACHE=/fs/clip-scratch/adesai10/hub
VLLM_CACHE=/fs/clip-scratch/adesai10/vllm
CONTAINER=/fs/clip-scratch/adesai10/containers/vllm-gemma4.sandbox
SERVER_SCRIPT=jobs/run_vllm_a6000.sbatch
```

`/fs/clip-scratch/adesai10/hub` is the persistent Hugging Face cache. Model
weights should live there, not in `/tmp` and not in the repo `.venv`.

The server script binds:

```text
/fs/clip-scratch/adesai10 -> /root/.cache/huggingface
/fs/clip-scratch/adesai10/vllm -> /root/.cache/vllm
```

So inside the container, Hugging Face sees:

```text
HF_HOME=/root/.cache/huggingface
HF_HUB_CACHE=/root/.cache/huggingface/hub
```

On the host, the same files are under:

```text
/fs/clip-scratch/adesai10/hub
```

## Mental Model

Think of the Apptainer job as a model server. Think of the Inspect jobs as API
clients.

Changing datasets, modes, shards, run names, question ranges, or `--max-tokens`
does not require restarting Apptainer.

Changing the served base model usually does require restarting the vLLM server.
A running vLLM server is normally started with exactly one base model:

```bash
--model google/gemma-4-E4B-it
```

To switch from Gemma to Qwen, stop that server job and start a new server with
the Qwen model. You do not reinstall vLLM for each model if the container
already supports that model family. You only need the model weights in the cache.

## Step 1: Choose Quantized Weights That Fit on 1 A6000

An RTX A6000 has 48 GB of GPU memory. The model must fit model weights plus KV
cache plus vLLM overhead in that 48 GB.

The rough memory rule:

```text
GPU memory = model weights + KV cache for max context/concurrency + runtime overhead
```

Approximate weight sizes:

| Weight type | Approx bytes per parameter | Example 27B weight size |
|---|---:|---:|
| BF16 / FP16 | 2 bytes | about 54 GB |
| FP8 | 1 byte | about 27 GB |
| INT4 / AWQ / GPTQ | 0.5 bytes plus scales | about 14-18 GB |

This means:

- A 4B or 8B model usually fits comfortably on 1 A6000.
- A 14B BF16 model may fit, but leaves less room for long context.
- A 26B/27B BF16 model usually does not fit on 1 A6000.
- A 26B/27B FP8 or INT4 quantized model may fit, depending on context length.
- Long context is expensive because the KV cache grows with `--max-model-len`.

For one A6000, start conservatively:

```bash
VLLM_MAX_MODEL_LEN=32768
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_TENSOR_PARALLEL_SIZE=1
```

If startup fails with CUDA OOM, reduce context first:

```bash
VLLM_MAX_MODEL_LEN=16384
```

If it still fails, use a smaller or more aggressively quantized model.

### Quantization Flags

Some quantized model repos are auto-detected by vLLM. Others need an explicit
flag. Common examples:

```bash
VLLM_EXTRA_ARGS="--quantization awq"
VLLM_EXTRA_ARGS="--quantization gptq"
VLLM_EXTRA_ARGS="--quantization fp8"
```

Use only the quantization mode that matches the weight repo. Do not point vLLM
at a BF16 repo and add `--quantization awq`; that does not magically quantize
the model. The downloaded repository itself must contain quantized weights.

Examples of model naming patterns that often indicate quantized weights:

```text
*-AWQ
*-GPTQ
*-FP8
*-Int4
*-W4A16
```

Always check the model card for the expected vLLM command.

## Step 2: Download Weights into the Persistent Cache

Do this from a login node or compute node. It does not require a GPU unless the
model repository has unusual custom code.

Set the cache variables:

```bash
export HF_HOME=/fs/clip-scratch/adesai10
export HF_HUB_CACHE=/fs/clip-scratch/adesai10/hub
export MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub
export HF_TOKEN=<your_huggingface_token_if_needed>
```

Download the model:

```bash
huggingface-cli download google/gemma-4-E4B-it
```

For a quantized model, download the quantized repo, not the original BF16 repo:

```bash
huggingface-cli download <org>/<quantized-model-repo>
```

Check disk usage:

```bash
du -sh /fs/clip-scratch/adesai10/hub/models--*
```

If you need to remove one model later:

```bash
rm -rf /fs/clip-scratch/adesai10/hub/models--ORG--MODEL
rm -rf /fs/clip-scratch/adesai10/hub/.locks/models--ORG--MODEL
```

For example:

```bash
rm -rf /fs/clip-scratch/adesai10/hub/models--google--gemma-4-E4B-it
rm -rf /fs/clip-scratch/adesai10/hub/.locks/models--google--gemma-4-E4B-it
```

## Step 3: Start a vLLM Apptainer Server for Testing

Use the repo's server script:

```bash
jobs/run_vllm_a6000.sbatch
```

It requests one RTX A6000 by default:

```bash
#SBATCH --gres=gpu:rtxa6000:1
```

It does not activate or modify the repo `.venv`.

Start a Gemma 4 E4B-it server:

```bash
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa

HF_TOKEN=$HF_TOKEN \
VLLM_MODEL=google/gemma-4-E4B-it \
VLLM_SERVED_MODEL_NAME=google/gemma-4-E4B-it \
VLLM_PORT=8000 \
VLLM_MAX_MODEL_LEN=32768 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
sbatch jobs/run_vllm_a6000.sbatch
```

For a quantized model:

```bash
HF_TOKEN=$HF_TOKEN \
VLLM_MODEL=<org>/<quantized-model-repo> \
VLLM_SERVED_MODEL_NAME=<org>/<quantized-model-repo> \
VLLM_PORT=8000 \
VLLM_MAX_MODEL_LEN=32768 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_EXTRA_ARGS="--quantization awq" \
sbatch jobs/run_vllm_a6000.sbatch
```

Use the quantization flag required by that repo. If the model card says vLLM
auto-detects the quantization, omit `VLLM_EXTRA_ARGS`.

Check the server job:

```bash
squeue -u $USER -o '%i %T %M %R %j %b %m'
```

Open the server log:

```bash
tail -f logs/slurm/vllm/vllm-a6000_<job_id>.log
```

The log prints the node and URL, for example:

```text
vLLM node: clip06.umiacs.umd.edu
vLLM URL: http://clip06.umiacs.umd.edu:8000/v1
```

Sanity-check the OpenAI-compatible API:

```bash
curl http://clip06.umiacs.umd.edu:8000/v1/models \
  -H "Authorization: Bearer local-dev-key"
```

You should see JSON listing the served model.

Then test one completion directly:

```bash
curl http://clip06.umiacs.umd.edu:8000/v1/chat/completions \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "Answer with only: ready"}],
    "max_tokens": 16
  }'
```

If this fails, fix the server before submitting Inspect jobs.

## Step 4: Run a Small Inspect Smoke Test

Use `evaluate` for a tiny manual smoke. This runs on the host using this repo's
`.venv`; it talks to the Apptainer vLLM server over HTTP.

```bash
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa

VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py evaluate \
    --run-name smoke_gemma4_e4b_apptainer \
    --generator qwen \
    --model vllm/google/gemma-4-E4B-it \
    --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
    --dataset-types arc_challenge \
    --settings model_from_scratch \
    --modes choices_only \
    --limit 5 \
    --max-connections 1 \
    --max-tokens 512
```

For full-question smoke:

```bash
VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py evaluate \
    --run-name smoke_gemma4_e4b_full_question \
    --generator qwen \
    --model vllm/google/gemma-4-E4B-it \
    --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
    --dataset-types gpqa \
    --settings model_from_scratch \
    --modes full_question \
    --limit 5 \
    --max-connections 1 \
    --max-tokens 2048
```

For smoke tests, keep `--max-tokens` small. A 32k output cap can make failed or
verbose samples take a long time.

## Step 5: Submit Full Evaluation Slices Against Apptainer

For full cluster evaluation, use `submit-evaluate-cluster` and pass
`--model-base-url`.

This is the crucial detail: when the model starts with `vllm/` and
`--model-base-url` is set, this repo schedules the slice jobs as API-client
jobs. They do not request their own GPUs. The A6000 is reserved only by the
separate Apptainer server job.

Example full run for Gemma 4 E4B-it:

```bash
RUN_NAME=eval_qwen_gemma4_e4b_apptainer_$(date +%Y%m%d_%H%M%S)

VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py submit-evaluate-cluster \
    --run-name "$RUN_NAME" \
    --generator qwen \
    --models vllm/google/gemma-4-E4B-it \
    --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
    --dataset-types arc_challenge,gpqa,mmlu_pro \
    --settings model_from_scratch \
    --modes choices_only,full_question \
    --questions-per-job 100 \
    --max-connections 1 \
    --max-tokens 32768 \
    --partition clip \
    --account clip \
    --qos high \
    --mem 32G \
    --cpus-per-task 4
```

For a debug cluster run:

```bash
RUN_NAME=debug_gemma4_e4b_apptainer_$(date +%Y%m%d_%H%M%S)

VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py submit-evaluate-cluster \
    --run-name "$RUN_NAME" \
    --generator qwen \
    --models vllm/google/gemma-4-E4B-it \
    --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
    --dataset-types gpqa \
    --settings model_from_scratch \
    --modes full_question \
    --limit 20 \
    --questions-per-job 20 \
    --max-connections 1 \
    --max-tokens 2048
```

The finalizer is submitted automatically. It waits for the slice jobs and then
materializes the collected evaluated dataset from the `.eval` logs.

### Do Not Put Multiple vLLM Models in One Command Unless They Share One Server

With Apptainer, this option:

```bash
--model-base-url http://clip06.umiacs.umd.edu:8000/v1
```

applies to the whole submitted command. If you write:

```bash
--models vllm/model-a,vllm/model-b
--model-base-url http://clip06.umiacs.umd.edu:8000/v1
```

then both model-a and model-b slices will send requests to the same server URL.
That is only correct if that one server can serve both model names, which is not
the normal vLLM setup here.

The safer pattern is one submitted run per Apptainer server:

```text
server for model-a -> submit run with --models vllm/model-a and server-a URL
server for model-b -> submit run with --models vllm/model-b and server-b URL
```

Use distinct run names so the outputs and finalizers do not get mixed together.

## Reserving Non-A6000 Nodes for Slices

When using Apptainer, the A6000 GPU is already occupied by the vLLM server. The
Inspect slices should not reserve A6000s. They are just HTTP clients.

The repo handles this when both conditions are true:

```text
model starts with vllm/
--model-base-url is provided
```

Then the generated manifest marks those tasks as `api`, not `local`, and the
generated slice wrappers do not request GPUs.

Good:

```bash
--models vllm/google/gemma-4-E4B-it
--model-base-url http://clip06.umiacs.umd.edu:8000/v1
```

Risky:

```bash
--models vllm/google/gemma-4-E4B-it
# no --model-base-url
```

Without `--model-base-url`, the scheduler treats `vllm/...` as a local model
and may request GPUs for each slice.

You can confirm before submitting by writing the bundle only:

```bash
./.venv/bin/python main.py submit-evaluate-cluster \
  --run-name dryrun_gemma4 \
  --generator qwen \
  --models vllm/google/gemma-4-E4B-it \
  --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
  --dataset-types gpqa \
  --limit 5 \
  --write-only
```

Then inspect the generated manifest:

```bash
find jobs/generated/evaluate/dryrun_gemma4 -name manifest.json -print
```

Look for:

```json
"resource_class": "api"
```

### What `--gpu-count` Means Here

In `submit-evaluate-cluster`, `--gpu-count` is used as a scheduler concurrency
cap per resource class. For ordinary local vLLM jobs, that roughly corresponds
to how many GPU slice jobs should run at once. For Apptainer-backed vLLM with
`--model-base-url`, the slices are `api` jobs, so `--gpu-count` caps how many API
slice jobs are allowed to run concurrently.

It does not allocate GPUs for the API slices. The only GPU allocation is the
separate Apptainer server job.

For a single A6000 vLLM server, a conservative full-run submission can include:

```bash
--gpu-count 1
--max-connections 1
```

This serializes slice jobs at the SLURM level and also keeps each Inspect worker
to one in-flight model request. If you omit `--gpu-count`, many API slice jobs
can be submitted at once; that can still work, but they will all contend for the
same vLLM server.

## Multiple Apptainer Servers at the Same Time

You can run multiple vLLM Apptainer servers at the same time, usually one per
model or one per replica of the same model.

Each server needs its own GPU allocation and its own port.

Example: two servers on two A6000 jobs:

```bash
HF_TOKEN=$HF_TOKEN \
VLLM_MODEL=google/gemma-4-E4B-it \
VLLM_SERVED_MODEL_NAME=google/gemma-4-E4B-it \
VLLM_PORT=8000 \
sbatch --job-name=vllm-gemma-e4b-p8000 jobs/run_vllm_a6000.sbatch

HF_TOKEN=$HF_TOKEN \
VLLM_MODEL=<org>/<qwen-quantized-repo> \
VLLM_SERVED_MODEL_NAME=<org>/<qwen-quantized-repo> \
VLLM_PORT=8001 \
VLLM_EXTRA_ARGS="--quantization awq" \
sbatch --job-name=vllm-qwen-awq-p8001 jobs/run_vllm_a6000.sbatch
```

If they land on the same node, they are distinguished by port:

```text
http://clip06.umiacs.umd.edu:8000/v1
http://clip06.umiacs.umd.edu:8001/v1
```

If they land on different nodes, they are distinguished by hostname and port:

```text
http://clip06.umiacs.umd.edu:8000/v1
http://clip07.umiacs.umd.edu:8000/v1
```

Slices connect to the correct server only because you pass the correct
`--model-base-url` at submission time. There is no automatic service discovery.

The model name must also line up with the name the server accepts. If Inspect
runs with:

```bash
--models vllm/google/gemma-4-E4B-it
```

then the OpenAI request model is effectively:

```text
google/gemma-4-E4B-it
```

Start vLLM with the same served model name:

```bash
VLLM_MODEL=google/gemma-4-E4B-it
VLLM_SERVED_MODEL_NAME=google/gemma-4-E4B-it
```

If you intentionally use an alias, make sure `VLLM_SERVED_MODEL_NAME` matches
what Inspect will request after the `vllm/` prefix is removed.

Example Gemma run pinned to server A:

```bash
VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py submit-evaluate-cluster \
    --run-name eval_gemma_server_a \
    --generator qwen \
    --models vllm/google/gemma-4-E4B-it \
    --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
    --dataset-types arc_challenge,gpqa,mmlu_pro \
    --questions-per-job 100 \
    --max-connections 1 \
    --max-tokens 32768
```

Example Qwen run pinned to server B:

```bash
VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py submit-evaluate-cluster \
    --run-name eval_qwen_server_b \
    --generator qwen \
    --models vllm/<org>/<qwen-quantized-repo> \
    --model-base-url http://clip07.umiacs.umd.edu:8000/v1 \
    --dataset-types arc_challenge,gpqa,mmlu_pro \
    --questions-per-job 100 \
    --max-connections 1 \
    --max-tokens 32768
```

### What If One Server Finishes Early?

vLLM server jobs do not naturally "finish" after a slice completes. They run
until one of these happens:

- you cancel the SLURM job,
- the job hits its wall-clock time limit,
- vLLM crashes,
- the node fails.

If server A is canceled or times out, only slices pointed at server A's
`--model-base-url` are affected. Slices pointed at server B continue normally.

The finalizer for a run depends on that run's slice jobs, not on the Apptainer
server job. If a server dies and its slices fail, the finalizer may still run
after the failed slice jobs exit, but the collected dataset will reflect the
failed or missing samples. Check slice logs and `.eval` status before trusting
the output.

### Can One Run Use Multiple Servers?

For one model, this repo's `submit-evaluate-cluster` command accepts one
`--model-base-url` for the submitted run. The simple pattern is therefore:

```text
one submitted run -> one vLLM server URL
```

If you want to use multiple replicas of the same model for throughput, submit
separate run names or separate question ranges, each with a different
`--model-base-url`, then combine outputs intentionally after checking that the
runs used the same model, generator, settings, modes, dataset version, and
question ranges.

Avoid accidentally submitting two jobs with the same run name to two different
servers unless you are deliberately managing non-overlapping slices. It makes
debugging and collection harder.

## Concurrency and GPU Limits

There are two different concurrency layers:

1. SLURM slice concurrency: how many Inspect jobs are running.
2. vLLM request concurrency: how many requests those jobs send to the server.

For 1 A6000 and a large/quantized model, use:

```bash
--max-connections 1
```

That means each Inspect process sends one request at a time. If you submit many
slices, they may still all connect to the same vLLM server, but the server will
queue and schedule requests according to available GPU memory and KV cache.

Start with these safe settings:

```bash
--questions-per-job 50
--max-connections 1
--max-tokens 2048     # smoke or first full-question debug
```

For full runs after the model is stable:

```bash
--questions-per-job 100
--gpu-count 1
--max-connections 1
--max-tokens 32768
```

If the server is underutilized, increase `--questions-per-job` or allow more
slices to run. If the server OOMs or becomes unstable, reduce:

```bash
VLLM_MAX_MODEL_LEN
--max-tokens
number of simultaneous submitted slices
```

Do not increase `VLLM_GPU_MEMORY_UTILIZATION` above `0.90` unless you are
debugging carefully. Leaving some GPU memory unused helps avoid fragmentation
and startup failures.

## Combining and Collection

The normal cluster path still works with Apptainer.

`submit-evaluate-cluster` writes:

```text
jobs/generated/evaluate/<run-name>/...
```

It submits:

```text
per-slice eval jobs
one finalizer job
```

The slice jobs run Inspect with `--skip-collect-evaluated`. The finalizer runs
after the slices and materializes the collected evaluated dataset.

You can monitor:

```bash
squeue -u $USER -o '%i %T %M %R %j %b %m'
```

Check accounting:

```bash
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList -P
```

Find generated jobs and manifests:

```bash
find jobs/generated/evaluate/<run-name> -maxdepth 4 -type f | sort
```

Find Inspect logs:

```bash
find results/inspect/evaluation/<run-name> -name '*.eval' -print
```

The finalizer is independent of Apptainer except that the slice jobs need the
server to be alive while they are generating model outputs.

## Recommended End-to-End Workflow

Use this sequence for a new quantized model.

### 1. Download weights

```bash
export HF_HOME=/fs/clip-scratch/adesai10
export HF_HUB_CACHE=/fs/clip-scratch/adesai10/hub
export HF_TOKEN=<token-if-needed>

huggingface-cli download <org>/<quantized-model-repo>
du -sh /fs/clip-scratch/adesai10/hub/models--*
```

### 2. Start vLLM on 1 A6000

```bash
HF_TOKEN=$HF_TOKEN \
VLLM_MODEL=<org>/<quantized-model-repo> \
VLLM_SERVED_MODEL_NAME=<org>/<quantized-model-repo> \
VLLM_PORT=8000 \
VLLM_MAX_MODEL_LEN=32768 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_EXTRA_ARGS="--quantization awq" \
sbatch jobs/run_vllm_a6000.sbatch
```

If no quantization flag is needed:

```bash
HF_TOKEN=$HF_TOKEN \
VLLM_MODEL=<org>/<quantized-model-repo> \
VLLM_SERVED_MODEL_NAME=<org>/<quantized-model-repo> \
VLLM_PORT=8000 \
VLLM_MAX_MODEL_LEN=32768 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
sbatch jobs/run_vllm_a6000.sbatch
```

### 3. Wait for the server URL

```bash
tail -f logs/slurm/vllm/<server_job_name>_<job_id>.log
```

Example:

```text
vLLM URL: http://clip06.umiacs.umd.edu:8000/v1
```

### 4. API sanity check

```bash
curl http://clip06.umiacs.umd.edu:8000/v1/models \
  -H "Authorization: Bearer local-dev-key"
```

### 5. Tiny Inspect smoke

```bash
VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py evaluate \
    --run-name smoke_quantized_model \
    --generator qwen \
    --model vllm/<org>/<quantized-model-repo> \
    --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
    --dataset-types arc_challenge \
    --settings model_from_scratch \
    --modes choices_only \
    --limit 5 \
    --max-connections 1 \
    --max-tokens 512
```

### 6. Full cluster run

```bash
RUN_NAME=eval_quantized_model_$(date +%Y%m%d_%H%M%S)

VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py submit-evaluate-cluster \
    --run-name "$RUN_NAME" \
    --generator qwen \
    --models vllm/<org>/<quantized-model-repo> \
    --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
    --dataset-types arc_challenge,gpqa,mmlu_pro \
    --settings model_from_scratch \
    --modes choices_only,full_question \
    --questions-per-job 100 \
    --max-connections 1 \
    --max-tokens 32768 \
    --mem 32G \
    --cpus-per-task 4
```

## Troubleshooting

### `curl /v1/models` fails

Check that the server job is running:

```bash
squeue -j <server_job_id> -o '%i %T %M %R %j %b %m'
```

Check the server log:

```bash
tail -n 200 logs/slurm/vllm/<server_log>.log
```

Common causes:

- the model is still loading,
- the job is pending,
- vLLM crashed during startup,
- wrong hostname,
- wrong port,
- wrong API key,
- the compute node is not reachable from where you are running `curl`.

### vLLM fails with CUDA OOM at startup

Try:

```bash
VLLM_MAX_MODEL_LEN=16384
VLLM_GPU_MEMORY_UTILIZATION=0.85
```

If it still fails, use a smaller or more compressed quantized model.

### The model needs a chat template

Some models do not define a chat template in their tokenizer config. vLLM may
reject chat completions with an HTTP 400. If the model card provides a template,
pass it through `VLLM_EXTRA_ARGS`:

```bash
VLLM_EXTRA_ARGS="--chat-template /path/inside/container/chat_template.jinja"
```

The path must be visible inside the container. Files under
`/fs/clip-scratch/adesai10` are visible inside the container under
`/root/.cache/huggingface`.

### The server is alive but evals get zero score

Zero score does not always mean the server failed. Check sample completions in
the `.eval` logs. Possible causes:

- the model answered in an unexpected format,
- `--max-tokens` was too small,
- the prompt mode is too hard for the model,
- the model emitted reasoning but not a parseable final answer,
- the wrong model/server URL was used.

### Jobs are slow

Full-question GPQA with high `--max-tokens` can be slow. For debugging, use:

```bash
--limit 20
--questions-per-job 20
--max-tokens 2048
```

Only use `--max-tokens 32768` when you actually want to allow very long
outputs.

### Slices accidentally request GPUs

Make sure `--model-base-url` is present. For Apptainer-backed vLLM, the eval
slices should be API jobs, not local GPU jobs.

Dry-run and inspect the manifest:

```bash
./.venv/bin/python main.py submit-evaluate-cluster \
  --run-name dryrun_check \
  --generator qwen \
  --models vllm/google/gemma-4-E4B-it \
  --model-base-url http://clip06.umiacs.umd.edu:8000/v1 \
  --dataset-types arc_challenge \
  --limit 5 \
  --write-only

find jobs/generated/evaluate/dryrun_check -name manifest.json -print
```

The tasks should say:

```json
"resource_class": "api"
```

### The server times out before slices finish

Increase the server job time limit in `jobs/run_vllm_a6000.sbatch` or submit
with an override:

```bash
sbatch --time=24:00:00 jobs/run_vllm_a6000.sbatch
```

If a server job dies, slices currently connected to that URL fail or retry
according to Inspect settings. Other runs pointed at other servers are not
affected.

### Port conflicts

If another server is already using port 8000 on the same node, use a different
port:

```bash
VLLM_PORT=8001 sbatch jobs/run_vllm_a6000.sbatch
```

Then use:

```bash
--model-base-url http://<node>:8001/v1
```

## Operational Checklist

Before full runs:

- Confirm weights are in `/fs/clip-scratch/adesai10/hub`.
- Confirm the model is quantized enough for 1 A6000.
- Start exactly one vLLM server per model/replica.
- Give every server a unique hostname/port combination.
- Confirm `/v1/models` works.
- Run a 5-question smoke.
- Submit full slices with `--model-base-url`.
- Confirm generated tasks are API jobs if using Apptainer.
- Keep `--max-connections 1` for large models on 1 A6000.
- Make sure the server wall time is longer than the expected slice runtime.
- Check the finalizer and collected outputs before using the results.

## Commands Worth Remembering

List your jobs:

```bash
squeue -u $USER -o '%i %T %M %R %j %b %m'
```

Cancel a vLLM server:

```bash
scancel <server_job_id>
```

Inspect server memory after it finishes:

```bash
sacct -j <server_job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList -P
```

Tail server logs:

```bash
tail -f logs/slurm/vllm/<server_log>.log
```

Tail eval logs:

```bash
tail -f logs/slurm/evaluate/<run-name>/*.out
tail -f logs/slurm/evaluate/<run-name>/*.err
```

Check the vLLM API:

```bash
curl http://<node>:<port>/v1/models \
  -H "Authorization: Bearer local-dev-key"
```

Submit a server:

```bash
HF_TOKEN=$HF_TOKEN \
VLLM_MODEL=google/gemma-4-E4B-it \
VLLM_SERVED_MODEL_NAME=google/gemma-4-E4B-it \
VLLM_PORT=8000 \
VLLM_MAX_MODEL_LEN=32768 \
sbatch jobs/run_vllm_a6000.sbatch
```

Submit eval slices against that server:

```bash
VLLM_API_KEY=local-dev-key \
MODEL_CACHE_DIR=/fs/clip-scratch/adesai10/hub \
HF_HOME=/fs/clip-scratch/adesai10 \
  ./.venv/bin/python main.py submit-evaluate-cluster \
    --run-name eval_example \
    --generator qwen \
    --models vllm/google/gemma-4-E4B-it \
    --model-base-url http://<node>:8000/v1 \
    --dataset-types arc_challenge,gpqa,mmlu_pro \
    --questions-per-job 100 \
    --max-connections 1 \
    --max-tokens 32768
