# Optional training-time RTC for PI0.5

PI0.5 supports action-prefix-conditioned flow matching as an optional training objective. Existing inference-time RTC remains the default RTC deployment mode. This change does not introduce new parameter shapes, so existing weights can initialize fine-tuning. A checkpoint must actually be fine-tuned with this objective before using its training-time RTC sampler.

The implementation supports `continuous_flow`, the existing continuous action loss schemas, and MEM observations. Combining it with `structured_temporal` discrete heads is rejected.

## Training

Add these overrides when starting a new fine-tuning run from `--policy.path=/path/to/checkpoint/pretrained_model`:

```bash
--policy.training_rtc_config.enabled=true \
--policy.training_rtc_config.simulated_delay=16 \
--policy.training_rtc_config.delay_distribution=uniform
```

The feature is disabled by default. `simulated_delay` is an **exclusive** upper bound. The default `16` uniformly samples prefix lengths 0–15, covering 0–300 ms at 50 Hz. This matches the measured latency range of the current deployment while leaving at least 35 actions in a 50-step chunk to predict. `exponential` remains available and samples with probability proportional to `exp(-d)`; use it only when short delays should dominate.

Near episode boundaries, sampled delays are shortened when necessary to avoid conditioning on padded actions and to leave at least one real target when available. This changes the effective delay distribution for short, padded chunks.

The repository wrapper exposes the same options:

```bash
./train_vla_pi05.sh \
  --base-policy=/path/to/compatible_checkpoint/pretrained_model \
  --training-rtc=true \
  --training-rtc-simulated-delay=16 \
  --training-rtc-delay-distribution=uniform \
  --dry-run=true
```

Supply the dataset, output directory, GPU/batch settings, and action/MEM settings appropriate to the checkpoint. The wrapper writes its configured action and observation schema into new runs; it does not preserve all checkpoint settings automatically. Use its dry run to inspect the resolved command. Direct `lerobot_train` invocation with `--policy.path` preserves checkpoint policy fields unless overridden.

Start a new output directory and optimizer for a new RTC fine-tuning stage. `--resume-checkpoint` restores an interrupted run, including its saved RTC settings and optimizer state. It does not convert an old non-RTC run into a new experiment.

The Kinetix recipe's eight epochs are specific to that reference experiment. A rough LeRobot budget is `ceil(8 * N_train / global_batch)` optimizer steps, where `global_batch = batch_per_gpu * world_size * gradient_accumulation_steps`. Held-out episodes, dataset sampling, replacement sampling, and incomplete batches affect this estimate.

## Objective and time convention

For sampled prefix length `d`, action index `i`, clean actions `a`, and noise `e`:

```python
prefix = i < d
token_time = 0.0 if prefix else sampled_time
x = token_time * e + (1 - token_time) * a
target_velocity = e - a
```

Each action token receives its own timestep embedding, including PI0.5 AdaRMS conditioning. Prefix actions are clean and their losses are excluded. Dataset padding is also excluded; existing continuous action loss weighting applies to the remaining suffix.

LeRobot uses `t=0` for clean actions and integrates from `t=1` to `t=0`. Kinetix labels the same clean endpoint `tau=1`; the conventions are related by `t=1-tau`. Copying its literal clean-time constant would be incorrect.

At inference, the prefix is clamped to the supplied action values at clean time on every denoising step and in the final output. Only the suffix evolves. Vision/language KV caching remains in use, and the sampler does not invoke inference-time RTC's guidance correction. This repository's existing inference RTC already uses a closed-form correction without an extra autograd pass, so VJP-based reference implementations do not establish a fixed speedup for this implementation.

## Deployment

The checkpoint selects its RTC algorithm automatically. A checkpoint with
`training_rtc_config.enabled=true` uses learned action-prefix conditioning;
all other checkpoints use the existing inference-time RTC algorithm. Deployment
does not expose a separate RTC-algorithm override.

Chunk scheduling remains selectable. `--chunk-scheduling-mode=RTC` enables
asynchronous chunk replacement and uses the checkpoint-selected RTC algorithm;
`--chunk-scheduling-mode=SYNC` executes a complete chunk and disables RTC
conditioning.

Keep the server's other required flags and checkpoint contract settings. Training mode requires a checkpoint with `training_rtc_config.enabled=true` and `discrete_action_training_mode=continuous_flow`. A requested delay outside the trained range is rejected. Changing only checkpoint metadata does not train the model.

The runtime setting is independent of the training objective:

```python
from lerobot.policies.rtc.configuration_rtc import RTCConfig

policy.config.rtc_config = RTCConfig(enabled=True, mode="training")
policy.init_rtc_processor()
actions = policy.predict_action_chunk(
    processed_batch,
    prev_chunk_left_over=normalized_remaining_chunk,
    inference_delay=delay_steps,
)
```

The remaining chunk starts at the current observation's action origin. Its first `delay_steps` actions will execute while inference runs. Supply actions in the same transformed, normalized model space used for training, with the correct reference frame and feature order. These are pending future actions, not past action history. The existing VLA server handles chunk alignment, B2 pose-delta/EE reanchoring, and normalization before passing the prefix to the policy.

For the first prediction without a previous chunk, pass a zero delay to obtain ordinary unconditioned sampling. Positive delays require a real prefix of sufficient length. The VLA server handles this initial case automatically. Guidance weight and soft prefix schedules apply only to inference mode. Fine-tuning and robot evaluation are required to measure task quality and latency; implementation tests alone do not establish those results.

## Validation

CPU tests exercise real small Gemma/SigLIP networks in float32 and bfloat16, forward/backward with gradient checkpointing, cached sampling, per-token AdaRMS, exact prefix preservation, and zero-delay equivalence. Additional regressions cover prefix/padding loss masks, gate-aware weighting, the existing inference RTC path, legacy checkpoint metadata, configuration round trips and CLI overrides, launcher resume behavior, and server mode selection/reanchoring. Full-size GPU fine-tuning, distributed training, and robot performance have not been measured for this feature.
