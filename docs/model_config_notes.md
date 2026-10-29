# Model config notes — `configs/nemo/multitask_nah_es.yaml`

Justification of every value in the small FastConformer-Transformer AED config
used by `scripts/train_nemo_multitask.py`, checked against the vendored NeMo
checkout (`nemo_speech/`, pinned at this commit — line numbers refer to it).

Verdicts:

- **ref** — copied verbatim from
  `nemo_speech/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`
  (the Canary-family recipe).
- **code** — value is set/consumed automatically by model source; the config
  just must not fight it.
- **ours** — deliberate deviation from NeMo; reason given, risk tracked in
  [Open deviations](#open-deviations-worth-acting-on-after-the-smoke-run).

## Model / prompting block

| Param | Value | Verdict | Justification |
|---|---|---|---|
| `sample_rate` | 16000 | ref | NeMo default; must match dataloader (`data/datamodule.py:16`). |
| `log_prediction` | true | ref | Logs sample predictions during training (ref comment). |
| `use_loss_mask_for_prompt` | false | ref | Canary convention: CE covers the full prompted transcript incl. prompt tokens; `true` masks the prompt (`aed_multitask_models.py:869-875`). |
| `prompt_format` | canary | ours (ref has `???`) | Ref demands explicit choice (`fast-conformer_aed.yaml:42`); `canary` v1 matches our manifests — `canary2` needs no_speech/timestamp/context slots we don't have (`prompts/canary.py:44-70`). |
| `multitask_metrics_cfg` (WER if `.source_lang==.target_lang`, BLEU otherwise, `13a`) | — | ref | Exact reference block (`fast-conformer_aed.yaml:47-58`); constraint syntax over `cut.custom` documented in `metrics/multitask.py:198-235`. Our script's manifest normalization fills these fields. |
| `train_ds/validation_ds/test_ds` | null | ours | Legal: ModelPT only instantiates `*_ds` if present **and** not null (`modelPT.py:154-158`); the external `CanaryMultilingualDataModule` supplies `PromptedAudioToTextMiniBatch`es directly. |
| `tokenizer.type: agg` + `custom_tokenizer: CanaryTokenizer` | — | ref | Canary prompting needs the aggregate (spl_tokens + per-lang); only `bpe`(=SentencePiece)/`wpe` sub-tokenizers are accepted (`mixins.py:202-253`; `canary_tokenizer.py:29-36, 165-193`). `langs` filled at runtime by the training script. |

## Encoder / decoder architecture

| Param | Value | Verdict | Justification |
|---|---|---|---|
| `model_defaults` 256/256/256, enc `n_layers: 12` | — | ours | NeMo publishes no small variants; closest in the ref table (`fast-conformer_aed.yaml:6-16`) is `canary-180m-flash` (enc 17L @512, dec 4L @1024). 12L@256 is the PLAN's smoke-run guess — flagged in the YAML header. |
| dec `num_layers: 4`, hidden 256 | — | ours | 4 decoder layers matches the *flash* family in the ref table (`:14`); hidden 256 is our guess. |
| equal enc/dec hidden sizes | — | code | If they differ a linear projection is auto-inserted; equal → `Identity` (`aed_multitask_models.py:163-166`; ref header `:7`). |
| `n_heads: 4` (enc & dec) | — | ours | Ref: heads "may need to be lower for smaller d_models" (`:175`); 256/4 = 64-dim heads. |
| `max_sequence_length: 512` | — | ref | Canary-1b row uses 512 (`:14`); safe since datamodule caps cuts at 40 s (`max_duration: 40.0`). |
| `subsampling: dw_striding`, `subsampling_factor: 8` | — | ref | FastConformer recipe (module default is `striding`/4x); factor must be a power of 2 (`conformer_encoder.py:77-80`). |
| `subsampling_conv_channels: 256` | — | ref | Docstring default `-1` = d_model, so for us it equals the default; ref pins 256 to cheapen subsampling in 1024-dim models (`conformer_encoder.py:81-82`). |
| `self_attention_model: rel_pos`, `att_context_size: [-1,-1]`, `pos_emb_max_len: 5000` | — | ref | FastConformer defaults; full bidirectional context is right for offline ASR (memory bounded by lhotse `quadratic_duration`). |
| `xscaling: false`, `untie_biases: true`, `conv_norm_type: batch_norm`, `dropout_pre_encoder: 0.1` | — | ref (also code defaults) | The FastConformer stack; `xscaling: false` deliberately overrides the module default `True` (`conformer_encoder.py:329-338`). |
| `dropout: 0.1`, `dropout_att: 0.1`, `dropout_emb: 0.0`, `ff_expansion_factor: 4` | — | ref | Verbatim (`conformer_encoder.py:322`). |
| `causal_downsampling: false`, `reduction_*: null/1` | — | ref | Non-streaming model, no extra reduction on top of 8x (`conformer_encoder.py:85-90`). |
| `transf_encoder.num_layers: 0` | — | ref | Optional bridge encoder; model skips instantiation at 0 (`aed_multitask_models.py:170-176`). |
| `pre_ln: true`, `pre_ln_final_layer_norm: true`, `hidden_act: relu`, dropouts 0.1, `num_token_types: 0`, `learn_positional_encodings: false` | — | ref | Verbatim pre-LN Transformer decoder block (`fast-conformer_aed.yaml:186-215`). |
| `transf_decoder.vocab_size` / `head.num_classes` | None | code | Auto-set at runtime, padded to a multiple of 8 (`aed_multitask_models.py:184`). |

## Head / decoding / loss / optimizer

| Param | Value | Verdict | Justification |
|---|---|---|---|
| `head` TokenClassifier (1 layer, relu, `use_transformer_init: true`) | — | ref (= code defaults) | Matches `TokenClassifierConfig` defaults (`token_classifier.py:33-38`); embedding weight-tying wired automatically (`aed_multitask_models.py:201`). |
| `decoding: beam, beam_size 1, len_pen 0.0, max_generation_delta 50` | — | ref | These are `MultiTaskDecodingConfig` defaults (`multitask_decoding.py:643, 660`) — in-training validation decodes greedily for speed; raise beam only in standalone eval. |
| `loss: SmoothedCrossEntropy`, `label_smoothing: 0.0`, `pad_id: null` | — | ref | `pad_id: null` is intentional — overridden at runtime with `tokenizer.pad_id` (`aed_multitask_models.py:226`); padding excluded from loss (`smoothed_cross_entropy.py:26-41`). |
| `optim: adamw, lr 3e-4, betas [0.9,0.98], weight_decay 1e-3` | — | ref | Canary values, but ref header warns they assume an effective batch of ~2K (`fast-conformer_aed.yaml:4`); our effective batch is far smaller → LR is a starting point to retune. |
| `sched: InverseSquareRootAnnealing, min_lr: 1e-6` | — | ref | Same as ref; also why epoch-based training is safe (no total-steps dependency). |
| `warmup_steps: 100` | ours | Ref's 2500 targets 100K steps; the script recomputes it as 5% of estimated total steps from `--max-epochs` (steps/epoch ≈ train seconds ÷ `--batch-duration`). |

## Preprocessor / SpecAugment

| Param | Value | Verdict | Justification |
|---|---|---|---|
| 128 mel feats, 0.025 window, 0.01 stride, hann, dither 1e-5 | — | ref | Verbatim Canary preprocessor block (`fast-conformer_aed.yaml:118-131`); module defaults (80 feats) are deliberately overridden. |
| `freq_masks: 2`, `freq_width: 27`, `time_width: 0.05` | — | ref | Verbatim (`fast-conformer_aed.yaml:143-148`). |
| `time_masks: 5` (ref: 10) | ours | Ref itself: "you may use lower time_masks for smaller models to have a faster convergence" (`:148`) — our case (small model, small corpus). |

## Open deviations worth acting on after the smoke run

1. **Tokenizer vocab 2000** (`--vocab-size` default) vs ref hint "recommend
   small vocab size of 128 or 256 when using 4x sub-sampling"
   (`fast-conformer_aed.yaml:111`) — with 8x subsampling the Canary family uses
   small vocabularies. On our tiny corpus SentencePiece
   (`hard_vocab_limit=false`) likely shrinks on its own; test 500–1000 and
   measure per-language token fertility (PLAN step F).
2. **12L/256 dims are unvalidated guesses.** If the smoke run converges,
   compare against the `canary-180m-flash` shape before scaling.
3. Ref (`fast-conformer_aed.yaml:8`) recommends **initializing FastConformer
   from an ASR/SSL checkpoint** for real runs; we start random-initialized on
   purpose (PLAN section D). The config is ready for `init_from_nemo_model`.

## Applies also to `configs/nemo/asr_nah.yaml`

The ASR-only prototype config (5 h Nahuatl corpus) is a full copy of this one;
verified diff is exactly `multitask_metrics_cfg`: WER with **no** `constraint`
(applies to all cuts, `metrics/multitask.py` `_no_constraint`) and no BLEU.
Do not replace it with `null` — the model default is unconstrained WER **+ BLEU**
(`aed_multitask_models.py:237-247`). Architecture stays identical on purpose so
the same shape can be promoted to multitask later; keep the two files in sync.

---

*Update this file whenever the YAML changes; keep it 1:1 with
`configs/nemo/multitask_nah_es.yaml`. Regenerate line refs against the pinned
`nemo_speech/` revision only.*
