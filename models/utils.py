import numpy as np

def to_batched_graphs(attention_window,input_ids,attention_mask):
    B = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    n_blocks = seq_len//(attention_window//2)-1
    local_window_adj = np.zeros([seq_len,seq_len])
    for i in range(n_blocks):
        start = i*attention_window//2
        end = start+attention_window
        local_window_adj[start:end,start:end] = 1
    g_list = []
    for i in range(B):
        src, dst = np.nonzero(local_window_adj)
        # edge_mask = [ True if attention_mask[i,s]!=0 and attention_mask[i,s]!=0 else False for s,d in zip(src,dst) ]
        # g = dgl.graph((src[edge_mask], dst[edge_mask]))
        g = dgl.graph((src, dst))
        g_list.append(g)
    batched_g = dgl.batch(g_list)
    return batched_g
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        # [B*N, 12, 64][1966080, 12, 64]
        return {  out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True) }
    return func

def mask_attention_score(edges):
    # Whether an edge has feature 1
    # return (edges.data['h'] == 1.).squeeze(1)
    edge_mask = (edges.src['mask']*edges.dst['mask']).unsqueeze(-1) #E,1
                                                    #edges.data['score']: [E,H,1]

    return {"score": edges.data['score'].masked_fill_(edge_mask==False,-10000)}

def convert_edge_feat_to_adj_form(edge_feat,n_nodes = 128):
    edge_feat=g.edata['score'] #[E,H]
    later_shape = edge_feat.shape[1:]

    former_shape = torch.Size([n_nodes,n_nodes]) 
    shape = former_shape+later_shape  #[BN,BN,H]
    src_list,dst_list = g.edges()[0], g.edges()[1]
    adj_form = torch.zeros(shape)
    for idx,(src,dst) in enumerate(zip(src_list,dst_list)):
        adj_form[dst,src] = edge_feat[idx]
    return adj_form.cpu()