import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
from torch import optim


def add_models_args(parser):
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')         
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability, and a tokenized input string. 
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, Encoder, Decoder, input_embedding, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.0, bidirect=False):  
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.Encoder = Encoder 
        self.Decoder = Decoder 
        self.input_embedding = input_embedding

    # def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        res = []
        for ex_i in range(len(test_data)):          # ex_i = index of example
            input_ex = test_data[ex_i].x_indexed     
            x_tensor = torch.tensor(input_ex, dtype=torch.long)
            x_tensor_emb = self.input_embedding.forward(x_tensor)   # get embedding
            x_tensor_emb = x_tensor_emb.unsqueeze(0)                 

            (output_Enc, context_mask_Enc, h_t_Enc) = self.Encoder.forward( embedded_words=x_tensor_emb, 
                input_lens=torch.tensor(np.array( [len(test_data[ex_i].x_tok)] ) , dtype=torch.int) )    
                # :embedded_words: [batch size x sent len x input dim] tensor
                # :input_lens: [batch size]-length vector containing the length of each input sentence
                # returns (output, context_mask, h_t) ; h_t is a tuple (h,c)
                # output_Enc is for attention

            h_t_Enc_reshaped = ( h_t_Enc[0].unsqueeze(0), h_t_Enc[1].unsqueeze(0) )    
            hidden_Dec = h_t_Enc_reshaped        # first hidden_Dec = last hidden_Enc
            input_Dec = torch.tensor( [self.output_indexer.index_of(SOS_SYMBOL)], dtype=torch.long )     

            y_pred = []
            for tk_i in range(65):  # tk_i = index of token in output_ex                
                output_Dec, hidden_Dec, log_probs_Dec = self.Decoder.forward(input_Dec.unsqueeze(0), hidden_Dec, output_Enc)  
                input_Dec = torch.tensor( [torch.argmax(log_probs_Dec)] ) 
                if input_Dec == torch.tensor( [self.output_indexer.index_of(EOS_SYMBOL)], dtype=torch.long ): 
                    break 
                y_pred.append( self.output_indexer.get_object(input_Dec.item()) )   
            
            res.append( [Derivation(test_data[ex_i], 1, y_pred)] )
        return res


    def encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with inp_lens_tensor lengths.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words are real and which ones are pad tokens), and the encoder final states (h and c tuple). only the ENCODER FINAL STATES are needed for the basic seq2seq model. enc_output_each_word is needed for attention, and enc_context_mask is needed to batch attention.

        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings



class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_emb_dim: int, hidden_size: int, bidirect: bool):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_emb_dim, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence -> tensor
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        output is only needed for attention, and context_mask is only used for batched attention.
        """

        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)

        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors      

        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output , batch_first=True)     
        max_length = max(input_lens.data).item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)



def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len. Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                for i in range(0, max_len)]
                for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                        for i in range(0, max_len)]
                        for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])



class RNNDecoder(nn.Module):
    def __init__(self, emb_dim, hidden_size, output_vocab_size, bidirect=False, embedding_dropout_rate=0.0):
        super(RNNDecoder, self).__init__()
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.embedding = EmbeddingLayer(emb_dim, output_vocab_size, embedding_dropout_rate)   
        self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=bidirect) 

        self.hidden2word = nn.Linear(hidden_size, output_vocab_size)
        self.hidden2word_attn = nn.Linear(hidden_size*2 , output_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)    


    def forward(self, input, hidden, output_Enc):
        embedded_input = self.embedding.forward(input)  
        output, hidden = self.rnn(embedded_input, hidden)   # hidden = (h,c)

        # alpha = torch.matmul( output_Enc.squeeze(), hidden[0].squeeze() ) # for un-batched 
        alpha = torch.matmul( output_Enc, hidden[0].permute(1,2,0) )
        alpha = self.log_softmax(alpha.permute(0,2,1))     # attn weight, softmax or logsoftmax

        # import torch.nn.functional as F
        # alpha = F.softmax(alpha.permute(0,2,1) , -1)

        attn_vec = torch.matmul( alpha, output_Enc )  # attn_vec = weighted ave of output_Enc         
        attn_hidden_concat = torch.cat( (attn_vec, hidden[0].permute(1,0,2)) , 2 )     
        
        log_probs = self.log_softmax( self.hidden2word_attn(attn_hidden_concat) )
        return output, hidden, log_probs    



def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set if need to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False) 
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)  
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    # train_data: has EOS, already indexed
    # all_train_input_data: has padding 

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    import time
    start_time = time.time() ; 
    print('======== Starting Time:', start_time)

    num_epochs = 100
    learning_rate = 0.001   # 0.01 
    batch_size = 4      # 4 6 8 16 32 
    emb_dim = 50        # 50 100 
    hidden_size = 128   # 128 256 
    embedding_dropout_rate = 0.0
    bidirect = False

    Encoder = RNNEncoder(emb_dim, hidden_size, bidirect=False)      # input_emb_dim = 50 ; hidden_size = 128
    Decoder = RNNDecoder(emb_dim, hidden_size, len(output_indexer), bidirect=False, embedding_dropout_rate=0.0)
    input_embedding = EmbeddingLayer(input_dim=emb_dim, full_dict_size=len(input_indexer), embedding_dropout_rate=0.0)      

    optimizer_Enc = optim.Adam(Encoder.parameters(), lr=learning_rate) # optim.SGD 
    optimizer_Dec = optim.Adam(Decoder.parameters(), lr=learning_rate) 
    optimizer_emb = optim.Adam(input_embedding.parameters(), lr=learning_rate) 

    indices = [i for i in range(len(train_data))]   # = list(range(len(train_data))) = idx of examples in train data

    for epoch in range(num_epochs):
        random.shuffle(indices)     
        epoch_loss = 0.0

        batchs = []     # list of indices of every batch 
        num_batch = len(indices) // batch_size     # num of whole batch
        batchs.append( indices[0 : batch_size] )
        for i in range(1, num_batch):
            batchs.append( indices[(i*batch_size) : (i*batch_size + batch_size)] )

        for ii, batch_i in enumerate(range(len(batchs))):      # batch_i = idx of a batch 
            batch = batchs[batch_i]     # batch = indices of 8 ex in 1 batch 

            Encoder.zero_grad()
            Decoder.zero_grad()
            input_embedding.zero_grad()

            input_ex = [all_train_input_data[j] for j in batch]     
            output_ex = [all_train_output_data[j] for j in batch]   # list of list; all_train_input_data[j] = list of indices of 1 ex 
            
            x_tensor = torch.tensor(input_ex, dtype=torch.long)
            x_tensor_emb = input_embedding.forward(x_tensor)         
            input_lens_ = torch.tensor(np.array( [len(train_data[k].x_tok) for k in batch]  ) , dtype=torch.int)

            (output_Enc, context_mask_Enc, h_t_Enc) = Encoder.forward( embedded_words=x_tensor_emb, input_lens= input_lens_)     
                # :embedded_words: [batch size x sent len x input dim] tensor
                # :input_lens: [batch size]-length vector containing the length of each input sentence
                # returns (output, context_mask, h_t) ; h_t is a tuple (h,c) ; output_Enc is for attention

            h_t_Enc_reshaped = ( h_t_Enc[0].unsqueeze(0), h_t_Enc[1].unsqueeze(0) )  
            hidden_Dec = h_t_Enc_reshaped        # first hidden_Dec = last hidden_Enc

            input_Dec = torch.tensor( [[output_indexer.index_of(SOS_SYMBOL)] for b in batch], dtype=torch.long )     
                # first input_Dec = "<SOS>" 

            loss = 0.0
            for tk_i in range(output_max_len):    # tk_i = index of token in output_ex  
                output_Dec, hidden_Dec, log_probs_Dec = Decoder.forward(input_Dec, hidden_Dec, output_Enc) 
                
                teacher_enforcing = True if random.random() < 0.8 else False             
                if teacher_enforcing: 
                    input_Dec = torch.LongTensor( [[ output_ex[b][tk_i] ] for b in range(batch_size)]  ) # input gold 
                else:   
                    input_Dec = torch.argmax(log_probs_Dec.squeeze(), dim=1).unsqueeze(1)       # input model predict
                    
                if np.any( (input_Dec.squeeze() == output_indexer.index_of(EOS_SYMBOL) )) :
                    input_Dec = torch.LongTensor( [[ output_ex[b][tk_i] ] for b in range(batch_size)]  )  # back to teacher enforcing

                NLL_Loss = nn.NLLLoss(ignore_index=0)   # index of PAD is 0 
                loss += NLL_Loss(log_probs_Dec.squeeze() , torch.LongTensor([ output_ex[b][tk_i ] for b in range(batch_size)]) )

            loss.backward()
            optimizer_Enc.step()
            optimizer_Dec.step()
            optimizer_emb.step()        
            epoch_loss += loss.item()

        print("total loss on epoch %i: %f" %(epoch, epoch_loss) , '--- time since start:' , (time.time()-start_time)/60 , 'min')   

        if (epoch > 40) and (epoch_loss < 40):
            print('training time: ' , (time.time() - start_time)/60, 'min')

            return Seq2SeqSemanticParser(Encoder, Decoder, input_embedding, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.0, bidirect=False) 
  
        end_time = time.time() 
        if (end_time - start_time) > 15*60 :       
            print(end_time)
            print('EXIT EARLY ----- training time: ' , (end_time - start_time)/60, 'min')
            return Seq2SeqSemanticParser(Encoder, Decoder, input_embedding, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.0, bidirect=False) 

    end_time = time.time() 
    print('training time: ' , (end_time - start_time)/60, 'min')

    return Seq2SeqSemanticParser(Encoder, Decoder, input_embedding, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.0, bidirect=False) 

