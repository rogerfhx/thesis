import torch 
import torch.nn as nn
import torch.nn.functional as F

from embedding import rotary_position_embedding

# Overview
# - Bidrectional attention transformer than attends to binder 
#   sequence and, if specified, multiple receptor sequences

class multi_head_attention():
    """
    multi head attention with rotary positional embedding
    """
    def __init__(self, input, num_head, d_model):
        self.input = input
        self.num_head = num_head
        self.num_tokens = input.shape[0]
        self.d_model = d_model 

        self.query = torch.rand((num_head, d_model, d_model))
        self.key = torch.rand((num_head, d_model, d_model))
        self.value = torch.rand((num_head, d_model, d_model))

        self.rope = rotary_position_embedding(self.query, self.key)


    def forward(self):
        layer_norm = nn.LayerNorm(self.d_model)
        layer_norm(self.input)
        multi_context_input = torch.empty((self.num_head, self.num_tokens, self.d_model))
        mapped_rope = torch.vmap(torch.vmap(self.rope.rotate(), in_dims=1), in_dims=1)
        rotated_query = mapped_rope(self.query)
        rotated_key = mapped_rope(self.key)
        for i in range(self.num_head):
            multi_context_input[i] = F.scaled_dot_product_attention(rotated_query[i], rotated_key[i], self.value[i])

        return multi_context_input



class feed_forward():
    def __init__(self, d_model:int):
        self.d_ffn = 4 * d_model
        self.nn = nn.Sequential(nn.LayerNorm(d_model), 
                                nn.Linear(d_model, self.d_ffn), nn.GELU(),
                                nn.Linear(self.d_ffn, d_model))

    def forward(self, input:torch.tensor):
        return self.nn(input)



class transformer_block():
    def __init__(self, input:torch.tensor, d_model:int, num_head:int):
        self.input = input
        self.d_model = d_model
        self.num_head = num_head

        self.multi_head_attention = multi_head_attention(input, num_head, d_model)
        self.ffn = feed_forward(d_model)


    def forward(self):
       attended_input = self.multi_head_attention.forward() + self.input
       output = self.ffn.forward(attended_input) + attended_input
       return output



class transformer_stack():
    def __init__(self, input, num_layer:int, d_model:int, num_head:int):
        self.input = input
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_head = num_head

        self.transformer_block = nn.Sequential(*(transformer_block(input, d_model, num_head) 
                                             for _ in range(num_layer)))

    def forward(self):
        return self.transformer_block(self.input)
        
