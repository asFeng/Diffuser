from torch import nn
import torch
import math
from dgl.nn.functional import edge_softmax
import dgl.function as fn
from models.diffuser_utils import *
from models.utils import *
class DiffuserSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DiffuserAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.self = DiffuserSelfAttention(config, layer_id)
        self.output = DiffuserSelfOutput(config)

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            g=g,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


class DiffuserSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) #B,N,H,D
        
        bool_mask = (attention_mask>=0 )
        g = g.local_var()
        g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
        g.ndata['q'] =  query_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['k'] =  key_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['v'] =  value_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        
        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]
        g.apply_edges(mask_attention_score)   #kq
        e = g.edata.pop('score') 
        g.edata['score'] = edge_softmax(g, e)
        g.edata['score']= nn.functional.dropout(g.edata['score'], p=self.dropout, training=self.training)
        
        g.ndata["h"] = g.ndata["v"]
        alpha = 0.1
        for _ in range(5):
            g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))
            g.apply_nodes(lambda nodes: {'h' : (1.0 - alpha) * nodes.data['h'] + alpha * nodes.data['v']})
            g.ndata['h']= nn.functional.dropout(g.ndata['h'], p=self.dropout, training=self.training)

        attn_output = g.ndata['h'] #BN,H,D
        attn_output = attn_output.reshape(batch_size, seq_len,  self.num_heads, self.head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs


