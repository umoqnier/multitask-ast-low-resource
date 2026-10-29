# Multitask Nahuatl-Spanish Speech Models with NVIDIA NeMo

*A repository-oriented roadmap for ASR and automatic speech translation*

## 1. Objective

Build a from-scratch, prompted attention encoder-decoder model that jointly
learns Nahuatl automatic speech recognition (ASR) and Nahuatl-to-Spanish
automatic speech translation (AST). The baseline is a Canary-style
NeMo `EncDecMultiTaskModel` with one shared `FastConformer` encoder and one
task-conditioned Transformer decoder.

| **Task** | **Input** | **Output** | **Primary metrics** |
| --- | --- | --- | --- |
| ASR | Nahuatl speech | Nahuatl transcription | WER, CER |
| AST | Nahuatl speech | Spanish translation | BLEU, chrF |

## 2. Design

Use one prompted attention encoder-decoder model:

```
Nahuatl audio
      |
      v
Audio preprocessor
      |
      v
SpecAugment during training
      |
      v
Shared FastConformer encoder
      |
      v
Transformer decoder
      |
      +--------------------------+
      |                          |
      v                          v
Nahuatl transcription      Spanish translation
<asr><nah><nah>            <ast><nah><es>
```

The initial architecture should use:

- A shared audio preprocessor
- A shared FastConformer encoder
- A shared autoregressive Transformer decoder
- Task and language prompt tokens
- A Nahuatl–Spanish tokenizer
- Task-specific evaluation metrics

This closely matches NeMo's Canary-style `EncDecMultiTaskModel`. Canary uses a FastConformer encoder, a Transformer decoder, token-based task control, and supports both ASR and AST.

The decoder should receive tokens that identify:

- The task
- The input language
- The output language
- Optionally, whether punctuation is expected

```
<bos> <src:nah> <task:asr> <tgt:nah> ...
<bos> <src:nah> <task:ast> <tgt:es> ...
```

### Shared decoder

Motivations for a shared decoder:

- Minimizes the number of task-specific components
- Supports future multilingual expansion
- Allows ASR to act as an auxiliary task for AST
- Follows NeMo's existing multitask model instead of requiring a custom model architecture
- Makes it possible to control inference explicitly through task tokens

NeMo's `EncDecMultiTaskModel` already uses prompted audio batches, a Transformer decoder, an aggregate tokenizer, and multitask metrics.

### Reuse previous manifests layout

Keep existing `data/datamodule.py` contract:

```
manifests/<lang>/<task>/
```

The data module remains responsible for:

- Resolving `manifests/<lang>/<task>`
- Loading the JSON manifests
- Selecting train, validation, and test splits
- Creating datasets
- Combining or sampling tasks
- Constructing data loaders
- Normalize all records to a canonical sample with audio_filepath, duration, text, utt_id, task, source_lang, target_lang, and punctuation mode.
- If task metadata is absent from JSON, infer it from the directory path rather than rewriting the corpus immediately.
- Validate paths, durations, audio readability, text, Unicode normalization, split leakage, and paired ASR/AST utterance IDs.

The NeMo-facing code should consume the loaders produced by your data module.

### NeMo paths to inspect
nemo/collections/asr/data/audio_to_text.py nemo/collections/common/parts/preprocessing/manifest.py


## 3. Target repository structure

```
multitask-ast-low-resource/
├── data/
│   ├── datamodule.py
│   ├── dataset.py
│   ├── collate.py
│   └── sampling.py
├── manifests/
│   └── nah/
│       ├── asr/
│       └── ast/
├── models/
│   ├── nemo_multitask_model.py
│   ├── batch_adapter.py
│   └── prompting.py
├── tokenizers/
│   └── nah_es/
│       ├── tokenizer.model
│       ├── tokenizer.vocab
│       └── tokenizer_config.yaml
├── configs/
│   └── nemo/
│       ├── multitask_nah_es.yaml
│       ├── model/
│       │   └── fastconformer_aed_small.yaml
│       ├── data/
│       │   └── nah_es.yaml
│       └── trainer/
│           └── single_gpu.yaml
├── scripts/
│   ├── train_nemo_multitask.py
│   ├── evaluate_nemo_multitask.py
│   ├── train_tokenizer.py
│   ├── inspect_batch.py
│   └── validate_manifests.py
├── tests/
│   ├── test_datamodule.py
│   ├── test_prompting.py
│   ├── test_batch_adapter.py
│   └── test_training_step.py
├── outputs/
├── datasets/
├── docs/
└── results/
```

## 4. Step by step implementation plan

### A. Pin the software environment

- Record the exact NeMo release or Git commit, CUDA, PyTorch, Lightning, Python, and container tag.

- Develop against that revision rather than a moving main branch.

- Commit a lock file or environment specification alongside every experiment.

```
environment:
  nemo_git_revision: "<commit-sha>"
  python: "3.12"
  cuda: "<version>"
  torch: "<version>"
```

#### NeMo paths to inspect

- README.md 
- pyproject.toml 
- uv.lock


### B. Run and study the Canary tutorial

#### NeMo reading order

| **Order** | **Resource** | **Repository path** | **What to learn** |
| --- | --- | --- | --- |
| 1 | Conceptual tutorial | nemo_speech/tutorials/asr/Canary_Multitask_Speech_Model.ipynb | Prompting, ASR/AST task control, inference |
| 2 | Training entry point | nemo_speech/examples/asr/speech_multitask/speech_to_text_aed.py | Hydra, Trainer, exp_manager, fit/test lifecycle |
| 3 | Reference config | nemo_speech/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml | FastConformer AED architecture and optimization |
| 4 | Core model | nemo_speech/nemo/collections/asr/models/aed_multitask_models.py | EncDecMultiTaskModel, steps, decoding, metrics |
| 5 | Prompted data | nemo_speech/nemo/collections/asr/data/audio_to_text_lhotse_prompted.py | Expected minibatch contract |
| 6 | Prompt formatters | nemo_speech/nemo/collections/common/prompts/ | Task/language prompt construction |
| 7 | Tokenizers | nemo_speech/nemo/collections/common/tokenizers/ | Canary and aggregate tokenizers |
| 8 | Decoding | nemo_speech/nemo/collections/asr/parts/submodules/multitask_decoding.py | Prompted AED decoding |
| 9 | Evaluation | nemo_speech/examples/asr/speech_to_text_eval.py; examples/asr/transcribe_speech.py | Standalone evaluation and inference |

For agents: `nemo_speech/.claude/skills`

Study the following concepts:

- [ ] How Canary distinguishes transcription from translation
- [ ] How source and target languages are specified
- [ ] How the prompt is assembled
- [ ] How the model selects an output language
- [ ] How tokenizers are organized
- [ ] How inference changes between ASR and AST
- [ ] What fields are expected in input manifests

#### output

Document (`docs/canary_notes.md`) that describe:

```
NeMo prompt → your task metadata
NeMo language ID → nah/es
NeMo manifest → your manifest
NeMo batch → your DataModule batch
```

### C. In deep NeMo's training entry point

`examples/asr/speech_multitask/speech_to_text_aed.py` should be my structural modelo for my own script. 

The upstream script trains an autoregressive speech AED model and accepts manifest paths, tokenizer directories, prompt format, model dimensions, and Lightning Trainer settings through configuration overrides. Focus on:

- Hydra configuration loading
- Trainer initialization
- exp_manager
- Model instantiation
- trainer.fit()
- Validation and test invocation
- Resume behavior
- Tokenizer configuration
- Distributed training parameters

Keep this script small. Data transformation, prompt construction, and task sampling should live in separate modules.

### D. Inspect the reference model configuration

Copy and reuse of `examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`. This configuration defines the autoregressive FastConformer-Transformer AED model used as the reference architecture for the Canary family. It also warns that its default optimization settings assume a very large effective batch and should be retuned for smaller batches.

#### output

`configs/nemo/multitask_nah_es.yaml`. We need to start with a substantially smaller model than Canary.

Example:

```yaml
model:
  sample_rate: 16000

  model_defaults:
    asr_enc_hidden: 256
    lm_enc_hidden: 256
    lm_dec_hidden: 256

  encoder:
    n_layers: 12
    d_model: 256
    n_heads: 4
    subsampling_factor: 8
    conv_kernel_size: 9
    dropout: 0.1

  transf_decoder:
    config_dict:
      hidden_size: 256
      num_layers: 4
      num_attention_heads: 4
      inner_size: 1024
      max_sequence_length: 512
```

> TODO: These values needs to be validated, not follows NeMo recommendations.

The upstream configuration describes billion-parameter Canary variants and explicitly **recommends initializing the FastConformer from an ASR or self-supervised encoder** for better convergence. Your first engineering validation should nevertheless use a small randomly initialized model because it is easier to debug.


### E. Define the task ontology and prompts

The NeMo AED architecture is identified by `EncDecMultiTaskModel`, a Transformer decoder, and prompt_format, so task prompting is a fundamental part of this model family rather than an optional preprocessing detail.

- Create `models/prompting.py` as the only registry for task and language mappings.
- Use explicit prompt tokens such as <src:nah>, <task:asr>, <task:ast>, <tgt:nah>, and <tgt:es>.

```python
TASKS = {
    "asr": {
        "source_lang": "nah",
        "target_lang": "nah",
        "task_token": "<task:asr>",
    },
    "ast": {
        "source_lang": "nah",
        "target_lang": "es",
        "task_token": "<task:ast>",
    },
}
```

- Begin with minimal prompts; add punctuation-control tokens only when the basic system works.

```
<bos> <src:nah> <task:asr> <tgt:nah> <nopnc>
<bos> <src:nah> <task:ast> <tgt:es> <pnc>
```

#### NeMo paths to inspect

```
nemo/collections/common/prompts/
nemo/collections/common/tokenizers/canary_tokenizer.py
tutorials/asr/Canary_Multitask_Speech_Model.ipynb
```


### F. Train and verify a joint tokenizer

- Train SentencePiece unigram or BPE jointly over Nahuatl ASR transcripts and Spanish AST targets.
- Start with 2,000 and 4,000 vocabulary variants; measure token fertility separately by language.
- Add BOS, EOS, PAD, language, task, and optional punctuation control tokens.

```
<pad>
<unk>
<bos>
<eos>
<src:nah>
<tgt:nah>
<tgt:es>
<task:asr>
<task:ast>
<pnc>
<nopnc>
```

- Assert that each control token maps to exactly one ID and that Nahuatl orthography and Spanish accents round-trip correctly.

> TODO: For multilingual do we need to retrain the tokenizer? Over which data?

#### NeMo paths to inspect

```
tutorials/asr/ASR_with_Subword_Tokenization.ipynb
tutorials/asr/Multilang_ASR.ipynb
nemo/collections/common/tokenizers/
nemo/collections/common/tokenizers/canary_tokenizer.py
```

NeMo distinguishes aggregate multilingual tokenization from ordinary single-tokenizer models. For the first implementation, a joint tokenizer is simpler. Later, compare it against separate Nahuatl and Spanish tokenizers using NeMo's aggregate tokenizer mechanism.
