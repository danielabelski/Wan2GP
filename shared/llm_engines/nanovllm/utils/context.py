from dataclasses import dataclass
import torch
from mmgp import cuda_graphs


@dataclass
class Context:
    is_prefill: bool = False
    has_previous_state: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    speculative_verify: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, has_previous_state=False, speculative_verify=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, bool(has_previous_state), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, bool(speculative_verify))

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


def _graph_key():
    # the Python values of the context that change what the blocks compute: part of the keys of their MMGP graphs
    context = _CONTEXT
    return context.is_prefill, context.has_previous_state, context.speculative_verify, context.block_tables is None, context.max_seqlen_q, context.max_seqlen_k


cuda_graphs.register_key(_graph_key)
