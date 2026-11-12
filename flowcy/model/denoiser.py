import torch

from embedding import token_embedding, segment_embedding, rotary_position_embedding, time_embedding

# Overview
# - Denoiser that combine embedding and transformer to predict the amino acid
#   logit from noisy protein sequence
# 
# - input: receptor Sequence <eos> binder sequence <eos> receptor sequence
# - output: 
#
# Architecture
# - embedding block
# - transformer block
# - MLP block: allow the denoiser to output logit


class embeddings_layer():
    def __init__(self, tokens, num_vocabs, d_model, is_conditional):
        self.tokens = tokens
        self.num_vocabs = num_vocabs
        self.d_model = d_model
        self.is_conditional = is_conditional

        self.token_embedding = token_embedding(tokens, num_vocabs, d_model)
        self.segment_embedding = segment_embedding(tokens, d_model, is_conditional)
        self.time_embedding = time_embedding()

    def forward(self, seq_token):
        token_embedding = self.token_embedding.embed(seq_token)
        segment_embedding = self.segment_embedding.embed(seq_token)
        if self.time_embedding:
            return token_embedding + segment_embedding + self.token_embedding.embed(seq_token)
        else:
            return token_embedding + segment_embedding


class output_layer():



class cytokine_model():

    def __init__(self):


    def forward(self):
        """
        embed --> transformer --> MLP
        """