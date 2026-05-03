"""
coreset/translate.py — Batch Cantonese translation via vLLM with TP + DP.

Parallelism layout
──────────────────
    total_gpus = tp_size × dp_size

    ┌─ Replica 0 ──────────────────┐  ┌─ Replica 1 ──────────────────┐
    │  GPU 0 │ GPU 1               │  │  GPU 2 │ GPU 3               │
    │  ←── tensor parallel (TP) ──→│  │  ←── tensor parallel (TP) ──→│
    └──────────────────────────────┘  └──────────────────────────────┘
              ↑  data parallel (DP) split  ↑
              └──────────── batch ─────────┘

Each replica is a Ray actor that owns a vLLM LLM(tensor_parallel_size=tp_size).
Placement groups pin each replica to its exclusive GPU bundle so CUDA devices
never overlap between replicas.

Example (4 × T4, tp=2, dp=2):
    translator = CantoneseTranslator(
        model="Qwen/Qwen3-4B",
        tensor_parallel_size=2,
        data_parallel_size=2,
    )
    translated = translator.translate_conversations(raw_texts, batch_size=64)
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

log = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你係一個專業翻譯員。"
    "請將以下文字翻譯成香港廣東話（繁體中文）。"
    "只需返回翻譯後嘅文字，唔需要任何解釋、前言或額外內容。"
)


def _build_prompt(text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Ray actor — one vLLM replica ──────────────────────────────────────────────

@ray.remote
class _TranslatorReplica:
    """
    A single vLLM inference replica. Instantiated inside a Ray placement-group
    bundle so it gets exactly ``tp_size`` GPUs and does not share them with any
    other replica.
    """

    def __init__(
        self,
        model: str,
        tp_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        temperature: float,
        max_new_tokens: int,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

    def translate(self, texts: List[str], max_num_seqs: int = 256) -> List[str]:
        """
        Translate a list of strings. vLLM processes them internally in steps
        of up to ``max_num_seqs`` prompts, so passing a large list is fine —
        GPU memory is managed by the engine, not the caller.
        """
        prompts = [_build_prompt(t, self.tokenizer) for t in texts]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]


# ── Translator ────────────────────────────────────────────────────────────────

class CantoneseTranslator:
    """
    Translates conversation turns (user / assistant) to Cantonese using vLLM.

    Supports both tensor parallelism (TP) and data parallelism (DP):

        tp_size  — GPUs per model replica (model sharding)
        dp_size  — number of independent replicas (batch sharding)

    Required GPUs = tp_size × dp_size.

    Args:
        model:                   HF model id (default: Qwen/Qwen3-4B).
        tensor_parallel_size:    GPUs to shard a single replica across.
        data_parallel_size:      Number of replicas to run in parallel.
        max_model_len:           Max token budget for prompt + generation.
        gpu_memory_utilization:  Fraction of VRAM allocated to vLLM.
        temperature:             Sampling temperature.
        max_new_tokens:          Max tokens generated per translation.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        max_model_len: int = 1_024,
        gpu_memory_utilization: float = 0.85,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ) -> None:
        self.dp_size = data_parallel_size
        self.tp_size = tensor_parallel_size

        gpus_needed = tensor_parallel_size * data_parallel_size
        print(
            f"[Translate] TP={tensor_parallel_size}  DP={data_parallel_size}  "
            f"total GPUs claimed={gpus_needed}"
        )

        # One placement-group bundle per replica, each bundle = tp_size GPUs.
        # STRICT_PACK keeps all bundles on the same node when possible; change
        # to "PACK" for multi-node clusters.
        bundles = [{"GPU": tensor_parallel_size, "CPU": 1}] * data_parallel_size
        self._pg = placement_group(bundles, strategy="PACK")
        ray.get(self._pg.ready())

        # Spawn one Ray actor per replica, bound to its bundle.
        actor_kwargs = dict(
            model=model,
            tp_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        self._replicas = [
            _TranslatorReplica.options(
                num_gpus=tensor_parallel_size,
                num_cpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_bundle_index=i,
                ),
            ).remote(**actor_kwargs)
            for i in range(data_parallel_size)
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _translate_flat(self, texts: List[str], max_num_seqs: int) -> List[str]:
        """
        Split ``texts`` into ``dp_size`` equal slices and dispatch each slice
        to one replica in a single remote call. All replicas work concurrently;
        vLLM handles internal batching (up to ``max_num_seqs`` per step).

        Example — 10 000 texts, dp=2:
            replica 0 gets texts[0    : 5 000]   → 1 generate() call
            replica 1 gets texts[5 000:10 000]   → 1 generate() call
            (both run in parallel via ray.get)
        """
        import math
        chunk_size = math.ceil(len(texts) / self.dp_size)

        futures = []
        slices  = []           # track (start, end) so we can reassemble in order
        for i, replica in enumerate(self._replicas):
            start = i * chunk_size
            end   = min(start + chunk_size, len(texts))
            if start >= len(texts):
                break
            slices.append((start, end))
            futures.append(replica.translate.remote(texts[start:end], max_num_seqs))

        results: List[List[str]] = ray.get(futures)   # blocks until all replicas done

        # Reassemble in original order
        flat: List[str] = []
        for batch_result in results:
            flat.extend(batch_result)
        return flat

    @staticmethod
    def _flatten(
        conversations: List[List[Dict[str, str]]],
        fields: Tuple[str, ...] = ("user", "assistant"),
    ) -> Tuple[List[str], List[Tuple[int, int, str]]]:
        flat_texts: List[str] = []
        flat_keys:  List[Tuple[int, int, str]] = []
        for conv_idx, turns in enumerate(conversations):
            for turn_idx, turn in enumerate(turns):
                for field in fields:
                    text = turn.get(field, "")
                    if text:
                        flat_texts.append(text)
                        flat_keys.append((conv_idx, turn_idx, field))
        return flat_texts, flat_keys

    # ── Public API ────────────────────────────────────────────────────────────

    def translate_conversations(
        self,
        all_conversations: List[List[Dict[str, str]]],
        max_num_seqs: int = 256,
        fields: Tuple[str, ...] = ("user", "assistant"),
    ) -> List[List[Dict[str, str]]]:
        """
        Translate the ``user`` and ``assistant`` fields to Cantonese.
        ``source`` and all other keys are left untouched.

        The flat list of texts is split evenly across ``dp_size`` replicas.
        Each replica receives one large chunk and calls ``llm.generate()``
        once — vLLM internally steps through it in batches of up to
        ``max_num_seqs`` prompts, keeping GPU utilisation high throughout.

        Args:
            all_conversations: List of examples, each a list of turn-dicts.
            max_num_seqs:      vLLM internal batch size (prompts per forward
                               pass). Default 256 works well for T4 16 GB;
                               raise for A100/H100.
            fields:            Turn-level keys to translate.
        """
        flat_texts, flat_keys = self._flatten(all_conversations, fields)

        print(
            f"[Translate] {len(flat_texts):,} segments  "
            f"({len(all_conversations):,} examples)  "
            f"→ Cantonese  [TP={self.tp_size} × DP={self.dp_size}  "
            f"max_num_seqs={max_num_seqs}]"
        )

        translated = self._translate_flat(flat_texts, max_num_seqs)

        result = copy.deepcopy(all_conversations)
        for (conv_idx, turn_idx, field), translation in zip(flat_keys, translated):
            result[conv_idx][turn_idx][field] = translation

        return result

    def shutdown(self) -> None:
        """Kill replica actors and release their placement group."""
        for replica in self._replicas:
            ray.kill(replica)
        remove_placement_group(self._pg)