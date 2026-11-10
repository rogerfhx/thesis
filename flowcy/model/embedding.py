import torch 
import torch.nn as nn


class token_embedding(nn.Module):
    """
    embed each token into d_model dimension vector
    """
    def __init__(self, tokens: torch.tensor, num_vocabs: int, d_model: int):
        super().__init__()
        self.tokens = tokens
        self.d_model = d_model
        self.embedding = nn.Embedding(num_vocabs, self.d_model)

    def embed(self) -> torch.tensor:
        return self.embedding(self.tokens)



class segment_embedding():
    """
    for binder receptor compatibility pre training objective
    """
    def __init__(self, num_seq, d_model):
        self.num_seq = num_seq
        self.d_model = d_model

    def embed(self):
        return 



class rotary_position_embedding():
    def __init__(self, query, key):
        self.query = query
        self.key = key


    def get_frequencies(self, token_position: int, d_model: int):
        """
        i in {1, 2, ..., d/2}
        """
        frequencies = torch.ones(d_model)
        i = torch.arange(d_model)
        i = torch.div(i, 2, rounding_mode="floor") - 1
        frequencies = token_position * (d_model) * ((1e4) ** (-2 * i / d_model))
        return frequencies


    def rotate(self, matrix:torch.tensor, token_position: int, d_model: int):
        """
        rotate the query and key vector using element wise multiplication
        rotation matrix = [mθ_1, ..., mθ_{d/2}]
        """
        frequencies = self.get_frequencies(token_position, d_model)

        indices = torch.arange(matrix.shape[0])
        indices_even_copy = indices[0::2].clone()
        indices[0::2] = indices[1::2]
        indices[1::2] = indices_even_copy 
        permutated_matrix = matrix.clone()[indices]

        permutated_matrix[0::2] *= -(torch.sin(frequencies.T))
        permutated_matrix[1::2] *= torch.sin(frequencies.T)

        rotated_matrix = matrix * torch.cos(frequencies.T) - permutated_matrix

        return rotated_matrix







    


# question
# - so segment embedding is learned during training, but what happen for different number of segment
# -