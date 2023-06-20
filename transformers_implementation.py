import torch #http://jalammar.github.io/illustrated-transformer/
import torch.nn as nn
import math
num_heads = 8
embed_len =512
batch_size = 8
stack_len = 6
dropout = 0.1

output_vocab_size = 7000
input_vocab_size = 7000

class InputEmbedding(nn.Module):
    def __init__(self, input_vocab_size = input_vocab_size, embed_len = embed_len, dropout = dropout,device ='cpu') :
        super(InputEmbedding, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_len = embed_len
        self.dropout = dropout
        self.device = device
        
        self.fisrtEmbedding = nn.Embedding(self.input_vocab_size,self.embed_len) # first embedding layer
        self.secondEmbedding = nn.Embedding(self.input_vocab_size,self.embed_len) # positional embedding layer
        
        self.dropoutLayer = nn.Dropout(p = self.dropout)
    
    
    def forward(self,input):
        first_embedding = self.fisrtEmbedding(input)
        batch_size, seq_len = input.shape
        
        positions_vector = torch.arange(0,seq_len).expand(batch_size,seq_len)
        positional_encoding = self.secondEmbedding(positions_vector)
        
        return self.dropoutLayer(first_embedding+ positional_encoding)
        

class ScaledDotProduct(nn.Module):
    def __init__(self, embed_len = embed_len, mask=None):
        super(ScaledDotProduct,self).__init__()
        self.embed_len = embed_len
        self.mask = mask
        self.dk = embed_len # dimension of keys and queries
        
        self.softmax = nn.Softmax(dim=3)
        
    def forward(self, queries,keys,values):
        compatibility = torch.matmul(queries, torch.transpose(keys,2,3))
        compatibility = compatibility / math.sqrt(self.dk)
        
        compatibility = self.softmax(compatibility)
        
        if self.mask is not None:
            compatibility = torch.tril(compatibility)
        
        return torch.matmul(compatibility, torch.transpose(values,1,2))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=num_heads, embed_len=embed_len, batch_size=batch_size, mask=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.embed_len = embed_len
        self.head_length = int(self.embed_len/self.num_heads)
        self.mask = mask
        self.concat_output = []

        # Q, K, and V have shape: (batch_size, seq_len, embed_len)
        self.q_in = self.k_in = self.v_in = self.embed_len

        # Linear layers take in embed_len as input 
        # dim and produce embed_len as output dim
        self.q_linear = nn.Linear(int(self.q_in), int(self.q_in))
        self.k_linear = nn.Linear(int(self.k_in), int(self.k_in))
        self.v_linear = nn.Linear(int(self.v_in), int(self.v_in))

        # Attention layer.
        if self.mask is not None:
            self.attention = ScaledDotProduct(mask=True) 
        else:
            self.attention = ScaledDotProduct()

        self.output_linear = nn.Linear(self.q_in, self.embed_len)

    def forward(self, queries, keys, values):

        # Query has shape: (batch_size, seq_len, num_heads, head_length)
        # Then transpose it: (batch_size, num_heads, seq_len, head_length)
        queries = self.q_linear(queries).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        queries = queries.transpose(1, 2)

        # Same for Key as for Query above.
        keys = self.k_linear(keys).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        keys = keys.transpose(1, 2)

        # Value has shape: (batch_size, seq_len, num_heads, head_length)
        values = self.v_linear(values).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)

        # 'sdp_output' here has size: 
        # (batch_size, num_heads, seq_len, head_length)
        sdp_output = self.attention.forward(queries, keys, values)

        # Reshape to (batch_size, seq_len, num_heads*head_length)
        sdp_output = sdp_output.transpose(1, 2).reshape(
            self.batch_size, -1, self.num_heads * self.head_length)

        # Return self.output_linear(sdp_output).
        # This has shape (batch_size, seq_len, embed_len)
        return self.output_linear(sdp_output)   
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_len = embed_len, dropout=dropout):
        super(EncoderBlock,self).__init__()
        self.embed_len = embed_len
        self.dropout = dropout
        self.multihead = MultiHeadAttention()
        self.firstNorm = nn.LayerNorm(self.embed_len)
        self.secondNorm = nn.LayerNorm(self.embed_len)
        self.dropoutLayer = nn.Dropout(p = self.dropout)
        
        self.feedForward = nn.Sequential(
            nn.Linear(self.embed_len,self.embed_len*4),
            nn.ReLU(),
            nn.Linear(self.embed_len*4,self.embed_len)
        )
        
    def forward(self,queries, keys, values):
        attention_output = self.multihead.forward(queries,keys,values)
        attention_output = self.dropoutLayer(attention_output)
        first_sublayer_output = self.firstNorm(attention_output + queries)
        ff_output = self.feedForward(first_sublayer_output)
        ff_output = self.dropoutLayer(ff_output)
        
        return self.secondNorm(ff_output+first_sublayer_output)
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_len = embed_len, dropout = dropout) -> None:
        super(DecoderBlock, self).__init__()
        self.embed_len = embed_len
        self.dropout = dropout
        
        self.maskedMultihead = MultiHeadAttention(mask=True)
        self.firstNorm = nn.LayerNorm(self.embed_len)
        self.dropoutLayer = nn.Dropout(p = self.dropout)
        
        self.encoderBlock = EncoderBlock()
    
    def forward(self, queries, keys, values):
        masked_multihead_output = self.maskedMultihead.forward(queries,keys,values)
        masked_multihead_output = self.dropoutLayer(masked_multihead_output)
        first_sublayer_output = self.firstNorm(masked_multihead_output + queries)
        
        return self.encoderBlock(first_sublayer_output, keys, values)
 
    
class Transformer(nn.Module):
    def __init__(self, embed_len = embed_len, stack_len = stack_len, output= output_vocab_size):
        super(Transformer,self).__init__()
        self.embed_len = embed_len
        self.stack_len = stack_len
        self.output= output
        
        self.embedding = InputEmbedding()
        self.encStack = nn.ModuleList(EncoderBlock() for i in range(self.stack_len))
        self.decStack = nn.ModuleList(DecoderBlock() for i in range(self.stack_len))
        
        self.finalLinear = nn.Linear(self.embed_len, self.output)
        self.softmax = nn.Softmax()
        
    def forward(self, test_input, test_output):
        enc_output = self.embedding.forward(test_input)
        
        for enc_layer in self.encStack:
            enc_output = enc_layer.forward(enc_output,enc_output,enc_output)
        
        dec_output = self.embedding(test_output)
        for dec_layer in self.decStack:
            dec_output = dec_layer.forward(dec_output,enc_output,enc_output)
            
        final_output = self.finalLinear(dec_output)
        
        return self.softmax(final_output)
    
    
input_tokens = torch.randint(10,(batch_size,20))
output_target = torch.randint(10,(batch_size,20))

transformer = Transformer()
transformer_output = transformer.forward(input_tokens,output_target)