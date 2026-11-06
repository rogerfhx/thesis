import torch 
import torch.nn as nn


class token_embedding():
    def __init__(self, num_token, d_model):
        self.num_token = num_token
        self.d_model = d_model

    def embed(self):
        return nn.embedding(self.num_token, self.d_model)



class segment_embedding():


class rotary_position_embedding():
    
