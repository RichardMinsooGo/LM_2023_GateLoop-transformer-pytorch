! pip install einops
! pip install beartype 

from math import pi, log

import torch
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from beartype import beartype
from beartype.typing import Literal, Union, Optional

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast(enabled = False)
def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim = -1)

    return out.type(dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding(Module):
    @beartype
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[
            Literal['lang'],
            Literal['pixel'],
            Literal['constant']
        ] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store('dummy', torch.tensor(0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent = False)

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        freqs = self.forward(self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset), seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, offset = offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    @beartype
    def get_scale(
        self,
        t: Tensor,
        seq_len: Optional[int] = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            exists(seq_len)
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim = -1)

        if should_cache:
            self.tmp_store('cached_scales', scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    @autocast(enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and \
            not self.learned_freq and \
            exists(seq_len) and \
            self.freqs_for != 'pixel'
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs

# taken from S5-pytorch repository
# https://github.com/i404788/s5-pytorch/blob/74e2fdae00b915a62c914bf3615c0b8a4279eb84/s5/jax_compat.py#L51-L134

# will be adapted to test out GateLoop on a small scale https://arxiv.org/abs/2311.01927

import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Tuple, Callable

# helper functions

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# Pytorch impl. of jax.lax.associative_scan
# made specifically for axis of 1 (sequence of tokens for autoregressive modeling)

def associative_scan(
    operator: Callable,
    elems: Tuple[Tensor, Tensor]
):
    num_elems = int(elems[0].shape[1])

    if not all(int(elem.shape[1]) == num_elems for elem in elems[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems]))

    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[1]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs of elements.

        reduced_elems = operator(
          [elem[:, :-1:2] for elem in elems],
          [elem[:, 1::2] for elem in elems])

        # Recursively compute scan for partially reduced tensors.

        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = operator(
                [e[:, :-1] for e in odd_elems],
                [e[:, 2::2] for e in elems])
        else:
            even_elems = operator(
                odd_elems,
                [e[:, 2::2] for e in elems])

        # The first element of a scan is the same as the first element
        # of the original `elems`.

        even_elems = [
          torch.cat([elem[:, :1], result], dim=1)
          for (elem, result) in zip(elems, even_elems)]

        return list(map(_interleave, even_elems, odd_elems))

    return _scan(elems)

def _interleave(a, b):
    a_axis_len, b_axis_len = a.shape[1], b.shape[1]
    output_axis_len = a_axis_len + b_axis_len

    if (a_axis_len == (b_axis_len + 1)):
        b = pad_at_dim(b, (0, 1), dim = 1)

    stacked = torch.stack([a, b], dim=2)
    interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)

    return interleaved[:, :output_axis_len]




from functools import partial

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# from rotary_embedding_torch import RotaryEmbedding

# from gateloop_transformer.associative_scan import associative_scan

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def Sequential(*modules):
    modules = list(filter(exists, modules))
    num_modules = len(modules)

    if num_modules == 0:
        return nn.Identity()
    elif num_modules == 1:
        return modules[0]

    return nn.Sequential(*modules)

# rms norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# norm wrappers

class PreNorm(Module):
    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class PostNorm(Module):
    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

# feedforward

def FeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# attention

class CausalFullAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        rotary_emb = False,
        add_swish_gating = False,
        data_dependent_rel_pos = False,
        frac_gradient_data_dependent_rel_pos = 0.5,
        softmax_normalize = None
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.softmax_normalize = default(softmax_normalize, not data_dependent_rel_pos)

        self.scale = dim_head ** -0.5

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.data_dependent_rel_pos = data_dependent_rel_pos
        self.frac_gradient_data_dependent_rel_pos = frac_gradient_data_dependent_rel_pos

        if data_dependent_rel_pos:
            self.to_a = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                Rearrange('b n (h d c) -> b h n d c', h = heads, c = 2)
            )

        self.to_gates = None

        if add_swish_gating:
            self.to_gates = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                nn.SiLU(),
                Rearrange('b n (h d) -> b h n d', h = heads)
            )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim)
        )

    def forward(
        self,
        x,
        ablate_complex = False,
        ablate_state_transition = False
    ):
        q, k, v = self.to_qkv(x)

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        q = q * self.scale

        if self.data_dependent_rel_pos and not ablate_state_transition:
            frac_gradient = self.frac_gradient_data_dependent_rel_pos

            a = self.to_a(x)

            # allow for data dependent relative position projection to change more slowly
            # alternative to using a lowered learning rate mentioned in paper

            a = a * frac_gradient + a.detach() * (1 - frac_gradient)

            a = torch.view_as_complex(a)

            if ablate_complex:
                a = a.real + 0.j

            magnitude, phase = a.abs(), a.angle()
            a = torch.polar(magnitude.sigmoid(), phase)

            a = rearrange(a, '... -> ... 1')
            a_cumprod = a.cumprod(dim = -2)

            a_cumprod_real = a_cumprod.real.clamp(min = 1e-10)
            a_cumprod_real_inverse = 1. / a_cumprod_real

            q, k = map(lambda t: rearrange(t, '... (d c) -> ... d c', c = 2), (q, k))

            q = q * a_cumprod_real
            k = k * a_cumprod_real_inverse

            q, k = map(lambda t: rearrange(t, '... d c -> ... (d c)'), (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)

        if self.softmax_normalize:
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim = -1)
        else:
            attn = sim.masked_fill(causal_mask, 0.)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        if exists(self.to_gates):
            out = out * self.to_gates(x)

        return self.to_out(out)

# data gated linear attention with "gateloop operator"

def gate_loop_operator(q, k, v, a):
    """
    the pseudocode in section 3.2 of the paper
    """

    kv = einsum('b n d, b n e -> b n d e', k, v)
    kv = kv + 0.j

    def binary_operator(a, b):
        a_i, kv_i = a
        a_j, kv_j = b
        return a_j * a_i, a_j * kv_i + kv_j

    _, kv = associative_scan(binary_operator, (a, kv))

    return einsum('b n d, b n d e -> b n e', q, kv.real)

class GateLoopedAttention(Module):
    def __init__(
        self,
        dim,
        heads = None,
        dim_inner = None,
        checkpoint_gate_looped_attn = True,
        add_swish_gating = True,
        sub_ln = False,
        frac_gradient_state_transition = 0.9
    ):
        super().__init__()
        self.frac_gradient_state_transition = frac_gradient_state_transition
        self.checkpoint_gate_looped_attn = checkpoint_gate_looped_attn

        dim_inner = default(dim_inner, dim)
        heads = default(heads, dim_inner)

        self.heads = heads
        assert (dim_inner % heads) == 0, f'dimension for gate looped attention {dim_inner} must be divisible by number of gate loop heads {heads}'

        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_a = nn.Sequential(
            nn.Linear(dim, heads * 2),
            Rearrange('b n (h c) -> (b h) n 1 1 c', h = heads, c = 2)
        )

        self.merge_heads = Rearrange('(b h) n d -> b n (h d)', h = heads)

        self.maybe_sub_ln = nn.LayerNorm(dim_inner) if sub_ln else nn.Identity()

        self.to_gates = None

        if add_swish_gating:
            self.to_gates = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                nn.SiLU()
            )

        self.to_out = nn.Linear(dim_inner, dim, bias = False) if dim_inner != dim or add_swish_gating else nn.Identity()

    def forward(
        self,
        x,
        ablate_complex = False,
        ablate_state_transition = False
    ):
        frac_gradient = self.frac_gradient_state_transition

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        a = self.to_a(x)
        a = a * frac_gradient + a.detach() * (1 - frac_gradient)

        a = torch.view_as_complex(a)

        if ablate_complex:
            a = a.real + 0.j

        if ablate_state_transition:
            a = torch.ones_like(a.real) + 0.j
        else:
            # activations for state transitions
            # sigmoid for magnitude, identity for phase

            magnitude, phase = a.abs(), a.angle()
            a = torch.polar(magnitude.sigmoid(), phase)

        need_backwards = any([t.requires_grad for t in (q, k, v, a)])

        fn = partial(checkpoint, gate_loop_operator) if need_backwards and self.checkpoint_gate_looped_attn else gate_loop_operator

        out = fn(q, k, v, a)

        out = self.merge_heads(out)

        out = self.maybe_sub_ln(out)

        if exists(self.to_gates):
            out = self.to_gates(x) * out

        return self.to_out(out)

# main class

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        checkpoint_gate_looped_attn = True,
        use_gate_looped_attn = True,
        gate_loop_heads = None,
        attn_add_swish_gating = True,
        dim_gate_looped_attn = None,
        attn_softmax_normalize = None,
        data_dependent_rel_pos = False,
        frac_gradient_state_transition = 0.9,
        ablate_complex = False,
        ablate_state_transition = False,
        rotary_emb = False,
        post_ln_norm = False,
        sub_ln = False
    ):
        super().__init__()
        self.ablate_complex = ablate_complex
        self.ablate_state_transition = ablate_state_transition

        self.token_emb = nn.Embedding(num_tokens, dim)

        layers = ModuleList([])

        layer_wrapper = PreNorm if not post_ln_norm else PostNorm

        for _ in range(depth):

            if use_gate_looped_attn:
                spatial_mixer = GateLoopedAttention(
                    dim = dim,
                    heads = gate_loop_heads,
                    dim_inner = dim_gate_looped_attn,
                    add_swish_gating = attn_add_swish_gating,
                    sub_ln = sub_ln,
                    checkpoint_gate_looped_attn = checkpoint_gate_looped_attn,
                    frac_gradient_state_transition = frac_gradient_state_transition
                )
            else:
                spatial_mixer = CausalFullAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    rotary_emb = rotary_emb,
                    add_swish_gating = attn_add_swish_gating,
                    softmax_normalize = attn_softmax_normalize,
                    data_dependent_rel_pos = data_dependent_rel_pos,
                    frac_gradient_data_dependent_rel_pos = frac_gradient_state_transition
                )

            channelwise_mixer = FeedForward(
                dim = dim,
                mult = ff_mult
            )

            layers.append(ModuleList([
                layer_wrapper(dim, spatial_mixer),
                layer_wrapper(dim, channelwise_mixer)
            ]))

        self.layers = ModuleList(layers)

        self.to_logits = Sequential(
            RMSNorm(dim) if not post_ln_norm else None,
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        return_loss = False,
        ablate_complex = None,
        ablate_state_transition = None
    ):
        ablate_complex = default(ablate_complex, self.ablate_complex)
        ablate_state_transition = default(ablate_state_transition, self.ablate_state_transition)

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(
                x,
                ablate_complex = ablate_complex,
                ablate_state_transition = ablate_state_transition
            )

            x = ff(x)

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)


n_dec_vocab = 4321
n_input = 567

model = Transformer(
    num_tokens = n_dec_vocab,
    dim = 624,
    depth = 6,
    use_gate_looped_attn = True
)

ids = torch.randint(0, n_dec_vocab, (1, n_input))
logits = model(ids) # (1, n_input, n_dec_vocab)
print(logits.shape)

