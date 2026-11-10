import torch 

# tokenization method: each amino acid in a sequence is consider as one token

VOCAB_SIZE = 29

standard_amino_acids = {"Alanine": {"3-letter": "Ala", "1-letter": "A"},
                        "Arginine": {"3-letter": "Arg", "1-letter": "R"},
                        "Asparagine": {"3-letter": "Asn", "1-letter": "N"},
                        "Aspartic Acid": {"3-letter": "Asp", "1-letter": "D"},
                        "Cysteine": {"3-letter": "Cys", "1-letter": "C"},
                        "Glutamic Acid": {"3-letter": "Glu", "1-letter": "E"},
                        "Glutamine": {"3-letter": "Gln", "1-letter": "Q"},
                        "Glycine": {"3-letter": "Gly", "1-letter": "G"},
                        "Histidine": {"3-letter": "His", "1-letter": "H"},
                        "Isoleucine": {"3-letter": "Ile", "1-letter": "I"},
                        "Leucine": {"3-letter": "Leu", "1-letter": "L"},
                        "Lysine": {"3-letter": "Lys", "1-letter": "K"},
                        "Methionine": {"3-letter": "Met", "1-letter": "M"},
                        "Phenylalanine": {"3-letter": "Phe", "1-letter": "F"},
                        "Proline": {"3-letter": "Pro", "1-letter": "P"},
                        "Serine": {"3-letter": "Ser", "1-letter": "S"},
                        "Threonine": {"3-letter": "Thr", "1-letter": "T"},
                        "Tryptophan": {"3-letter": "Trp", "1-letter": "W"},
                        "Tyrosine": {"3-letter": "Tyr", "1-letter": "Y"},
                        "Valine": {"3-letter": "Val", "1-letter": "V"}}


non_standard_amino_acid = {"Asparagine": {"3-letter": "Asx", "1-letter": "B"},
                           "Selenocysteine": {"3-letter": "Sec", "1-letter": "U"},
                           "Glutamic acid": {"3-letter": "Glx", "1-letter": "Z"},
                           "Ornithine": {"3-letter": "Orn", "1-letter": "O"},}


special_token = {"Seperation": {"3-letter": "Sep"},
                 "Unknown": {"3-letter": "Unk", "1-letter": "X"},
                 "Masked": {"3-letter": "Msk"},
                 "Padding": {"3-letter": "Pad"},
                 "Classifcaton": {"3-letter": "Cls"}}


tokens = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", 
          "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", 
          "B", "U", "Z", "O", "<SEP>", "<UNK>", "<MSK>", "<PAD>", "<CLS>"]



class tokenizer():
    def __init__(self, binder_seq: str, receptor_seq: str|None, is_complex: bool|None):
        self.binder_seq = binder_seq
        self.receptor_seq = receptor_seq
        self.is_complex = is_complex
        

    def encode(self) -> torch.tensor:
        """
        map token to token ID
        """
        seq_tokenID = torch.tensor([tokens.index(token) for token in self.binder_seq])
        return seq_tokenID


    def decode(self, output:torch.tensor) -> str:
        """
        map token ID to token
        """
        seq = "".join([tokens[i] for i in output])
        return seq


# note
# -- so positonal embedding, segment embedding, and token embedding are all
#    intilized randomly the beginning and are learned as part of the model
#    training. The model output is a context awared embedding
# -- process the receptor and binder sequence add <CLS> 
#

# problem
# -- the order of sequence
#


# question
# - do i just add random embedding in the beginning?
# - what does the classification token actually contian? the true label?