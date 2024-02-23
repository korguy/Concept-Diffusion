from dataclasses import dataclass
from typing import Literal

import logging
import torch as th
import torch.nn as nn
try:
    from diffusers.models.cross_attention import CrossAttention
except:  # Newer diffusers versions
    from diffusers.models.attention import Attention as CrossAttention
from typing import Optional, Tuple


STRUCT_ATTENTION_TYPE = Literal["extend_str", "extend_seq", "align_seq", "none"]

@dataclass
class KeyValueTensors(object):
    k: th.Tensor
    v: th.Tensor

class StructuredCrossAttention(CrossAttention):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: int = 0,
        struct_attention: bool = False,
    ) -> None:
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)
        self.struct_attention = struct_attention

    def struct_qkv(
        self,
        q: th.Tensor,
        context: Tuple[th.Tensor, KeyValueTensors],
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        assert len(context) == 2 and isinstance(context, tuple)
        uc_context = context[0]
        context_k = context[1].k
        context_v = context[1].v

        if isinstance(context_k, list) and isinstance(context_v, list):
            return self.multi_qkv(
                q=q,
                uc_context=uc_context,
                context_k=context_k,
                context_v=context_v,
                mask=mask,
            )
        elif isinstance(context_k, th.Tensor) and isinstance(context_v, th.Tensor):
            return self.heterogenous_qkv(
                q=q,
                uc_context=uc_context,
                context_k=context_k,
                context_v=context_v,
                mask=mask,
            )
        else:
            raise NotImplementedError

    def multi_qkv(
        self,
        q: th.Tensor,
        uc_context: th.Tensor,
        context_k: th.Tensor,
        context_v: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> None:
        assert uc_context.size(0) == context_k.size(0) == context_v.size(0)

        # true_bs = uc_context.size(0) * self.heads
        # kv_tensors = self.get_kv(uc_context)

        raise NotImplementedError

    def normal_qkv(
        self,
        q: th.Tensor,
        context: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        batch_size, sequence_length, dim = q.shape

        k = self.to_k(context)
        v = self.to_v(context)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        hidden_states = self._attention(q, k, v, sequence_length, dim)

        return hidden_states

    def heterogenous_qkv(
        self,
        q: th.Tensor,
        uc_context: th.Tensor,
        context_k: th.Tensor,
        context_v: th.Tensor,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        batch_size, sequence_length, dim = q.shape

        k = self.to_k(th.cat((uc_context, context_k), dim=0))
        v = self.to_v(th.cat((uc_context, context_v), dim=0))

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        hidden_states = self._attention(q, k, v, sequence_length, dim)

        return hidden_states

    def get_kv(self, context: th.Tensor) -> KeyValueTensors:
        return KeyValueTensors(k=self.to_k(context), v=self.to_v(context))

    def forward(
        self,
        x: th.Tensor,
        context: Optional[Tuple[th.Tensor, KeyValueTensors]] = None,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        q = self.to_q(x)

        if isinstance(context, tuple):
            assert len(context) == 2
            assert isinstance(context[0], th.Tensor)  # unconditioned embedding
            assert isinstance(context[1], KeyValueTensors)  # conditioned embedding

            if self.struct_attention:
                out = self.struct_qkv(q=q, context=context, mask=mask)
            else:
                uc_context = context[0]
                c_full_seq = context[1].k[0].unsqueeze(dim=0)
                out = self.normal_qkv(
                    q=q, context=th.cat((uc_context, c_full_seq), dim=0), mask=mask
                )
        else:
            ctx = context if context is not None else x
            out = self.normal_qkv(q=q, context=ctx, mask=mask)

        return self.to_out(out)

class CustomCrossAttnProcessor:
    def __init__(self, struct_attention):
        super().__init__()
        self.struct_attention = struct_attention

    def heterogenous_qkv(
        self,
        q: th.Tensor,
        uc_context: th.Tensor,
        context_k: th.Tensor,
        context_v: th.Tensor,
        mask: Optional[th.Tensor] = None,
        attn = None,
    ) -> th.Tensor:

        batch_size, sequence_length, dim = q.shape

        k = attn.to_k(th.cat((uc_context, context_k), dim=0))
        v = attn.to_v(th.cat((uc_context, context_v), dim=0))

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        # hidden_states = self._attention(q, k, v, sequence_length, dim)
        attention_probs = attn.get_attention_scores(q, k, mask)
        hidden_states = th.bmm(attention_probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states

    def struct_qkv(
        self,
        q: th.Tensor,
        context: Tuple[th.Tensor, KeyValueTensors],
        mask: Optional[th.Tensor] = None,
        attn = None,
    ) -> th.Tensor:

        assert len(context) == 2 and isinstance(context, tuple)
        uc_context = context[0]
        context_k = context[1].k
        context_v = context[1].v

        out = [self.heterogenous_qkv(
                q=q,
                uc_context=uc_context,
                context_k=context_k[i].unsqueeze(0),
                context_v=context_v[i].unsqueeze(0),
                mask=mask,
                attn=attn
            ) for i in range(context_k.size(0))
        ]
        tmp = [0, 0]
        for i in out:
            tmp[0] += i[0,:,:]
            tmp[1] += i[1,:,:]
        tmp[0] = tmp[0].unsqueeze(0) / len(out)
        tmp[1] = tmp[1].unsqueeze(0) / len(out)
        return th.cat(tmp, dim=0)



    def normal_qkv(
        self,
        q: th.Tensor,
        context: th.Tensor,
        mask: Optional[th.Tensor] = None,
        attn = None,
    ):
        batch_size, sequence_length, dim = q.shape

        k = attn.to_k(context)
        v = attn.to_v(context)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        # hidden_states = self._attention(q, k, v, sequence_length, dim)
        attention_probs = attn.get_attention_scores(q, k, mask)
        hidden_states = th.bmm(attention_probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        q = attn.to_q(hidden_states)
        
        if isinstance(encoder_hidden_states, tuple):
            if self.struct_attention:
                out = self.struct_qkv(q=q, context=encoder_hidden_states, mask=attention_mask, attn=attn)
            else:
                uc_context = encoder_hidden_states[0]
                c_full_seq = encoder_hidden_states[1].k[0].unsqueeze(dim=0)
                out = self.normal_qkv(q=q, context=th.cat((uc_context, c_full_seq), dim=0), mask=attention_mask, attn=attn)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            out = self.normal_qkv(q=q, context=encoder_hidden_states, mask=attention_mask, attn=attn)

        hidden_states = attn.to_out[0](out)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def replace_cross_attention(nn_module: nn.Module, name: str) -> None:
    attn_procs = {}
    cross_att_count = 0
    for name in nn_module.attn_processors.keys():
        struct_attention = True if "attn2" in name else False
        cross_att_count += 1
        attn_procs[name] = CustomCrossAttnProcessor(
            struct_attention=struct_attention
        )
    nn_module.set_attn_processor(attn_procs)
