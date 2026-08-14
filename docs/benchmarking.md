# Benchmark protocol

This document defines the measurement and reporting protocol used by the
repository.

## Hypothesis

For Food-101 partial fine-tuning with ViT-B/16 at 384 x 384 and batch size 128,
the combined accelerated profile (TF32/high float32 matmul precision,
`torch.compile`, and BF16 autocast) reduces epoch time relative to FP32 eager
execution without a material loss in Top-1/Top-5 accuracy.

The accuracy tolerance is a predeclared experimental parameter. For example,
a study may allow an absolute final Top-1 difference no greater than 0.5
percentage points; the appropriate threshold depends on the application.

## Controlled variables

The following variables remain identical:

- exact training/evaluation examples and class-to-index mapping;
- resize, augmentation, normalization and sampling order;
- SWAG E2E pretrained checkpoint;
- trainable encoder blocks, layer norm and classifier head;
- initial random seed for each paired repeat;
- batch size, workers, prefetch and memory-transfer settings;
- Adagrad hyperparameters, label smoothing and cosine schedule;
- number of train/evaluation batches and epochs;
- physical GPU, CPU, storage and background workload.

Only these treatment variables differ:

- highest versus high float32 matmul precision;
- TF32 disabled versus enabled;
- eager versus `torch.compile` execution;
- FP32 versus automatic mixed precision.

This is a combined-treatment experiment. It estimates the speedup of the
bundle, not the isolated contribution of each switch. A separate ablation would
add profiles one at a time (TF32 only, AMP only, and compile only) while holding
all other fields fixed.

## Reference protocol

```bash
food101-benchmark \
  --data-root data \
  --download \
  --epochs 3 \
  --repeats 2 \
  --discard-first-epochs 1
```

Five repeats provide stronger timing statistics. Accuracy studies use the full
20 epochs and multiple seeds and are therefore substantially more expensive.

## Isolation and ordering

Each profile trial runs in a new spawned process. This prevents these forms of
cross-trial leakage:

- an already initialized CUDA context/allocator;
- in-memory Dynamo/Inductor graph caches;
- model or optimizer state from a previous profile;
- random-generator state from a previous trial.

Repeat 1 uses the requested profile order. Repeat 2 reverses it, and this
alternation continues. It reduces—but cannot eliminate—bias from filesystem
cache, GPU temperature, clock behavior, and external activity.

Valid timing runs assume no competing GPU-heavy applications, stable power and
clock settings, and an unset `CUDA_LAUNCH_BLOCKING` environment variable.

## Timing definitions

CUDA work is asynchronous, so the engine calls `torch.cuda.synchronize()`
immediately before starting and before stopping each phase timer.

The timed training phase includes:

- fetching all timed batches from the DataLoader;
- host-to-device transfer;
- forward pass and loss;
- backward pass;
- optimizer updates;
- first-use graph compilation when applicable;
- metric computation.

It excludes model/optimizer construction and checkpoint writing. Evaluation is
timed separately and total epoch time is train plus evaluation time.

`cold_epoch_speedup` includes every epoch. `steady_epoch_speedup` discards the
first `N` epochs from each run, where `N` is
`--discard-first-epochs`. A complete result includes `N` and both ratios.

## Accuracy protocol

The repository evaluates the official Food-101 test split each epoch to match
the original notebook. This is acceptable for a performance demonstration when
hyperparameters are already frozen, but repeated tuning against those values
turns the test set into a validation set.

An unbiased model-quality protocol has five stages:

1. the official training set is split into training and validation partitions;
2. learning rate, augmentations, epoch, and precision tolerance are selected
   using validation only;
3. every experimental choice is frozen;
4. the final declared procedure is trained;
5. the official test split is evaluated once.

The `best_eval_top1` field is descriptive rather than an unbiased test estimate
when it has influenced checkpoint or hyperparameter selection.

## Reading `benchmark.json`

A valid report satisfies these conditions:

1. both profiles report the same GPU and software environment;
2. dataset sizes, batch caps, model weights and parameter counts match;
3. `cuda_launch_blocking` is absent/null;
4. the accelerated profile reports all three feature switches enabled;
5. cold and steady timings contain the intended number of epochs;
6. standard deviations are acceptable across repeats;
7. throughput and time ratios tell a consistent story;
8. final accuracy remains inside the declared tolerance.

Results whose configurations differ beyond the treatment variables are not
directly comparable.

## Original notebook confounders

The saved 27.71-minute and 13.43-minute epochs are not a controlled pair:

- the slower notebook sets `CUDA_LAUNCH_BLOCKING=1`;
- it uses no DataLoader workers, while the accelerated notebook uses four;
- its RandAugment plus ColorJitter pipeline differs from TrivialAugmentWide;
- the optimizer learning rate and weight decay differ;
- gradient clipping differs;
- the recorded PyTorch environments differ or are incompletely documented;
- only one accelerated epoch completed;
- the GPU model was not recorded;
- notebook code was edited out of execution order, so some output may be stale.

That evidence motivates the hypothesis but cannot attribute the speedup to the
three techniques. The controlled runner exists specifically to correct this.

## Result schema

| Field | Interpretation |
|---|---|
| `cold_epoch_seconds` | Mean and standard deviation across all epochs, including compilation |
| `steady_epoch_seconds` | Mean and standard deviation after discarded warm-up epochs |
| `train_samples_per_second` | Training throughput measured inside the synchronized phase |
| `final_eval_top1` | Final-epoch Top-1 accuracy across repeats |
| `final_eval_top5` | Final-epoch Top-5 accuracy across repeats |
| `peak_cuda_memory_bytes` | Maximum allocated CUDA memory for the profile trial |

A complete result is accompanied by exact hardware, software, repeats, epochs,
batch caps (if any), and the accuracy tolerance. Capped smoke-test measurements
are labeled separately from full-data results.
