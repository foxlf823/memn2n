import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

class MemN2N(nn.Module):
    """End-To-End Memory Network."""
    def __init__(self, use_cuda, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        hops=3, nonlin=None, encoding=position_encoding):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            nonlin: Non-linearity. Defaults to `None`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.
        """
        
        super(MemN2N, self).__init__()
        
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._nonlin = nonlin
        self._use_cuda = use_cuda
        
        self._encoding = autograd.Variable(torch.from_numpy(encoding(self._sentence_size, self._embedding_size)))
        if use_cuda:
            self._encoding = self._encoding.cuda()
        
        # nil word embedding     
        self._nil_word_slot = autograd.Variable(torch.zeros(1, self._embedding_size))
        if use_cuda:
            self._nil_word_slot = self._nil_word_slot.cuda()
        
        # They are using as lookup tables instead of nn.Embedding
        # in the paper, if using the adjacent mode, B = A_1, C_k = A_(k+1), C_K = W
        # so if hops=3, we only need to define A_1 (B), C_1 (A_2), C_2 (A_3), C_3 (W)
        self.A_1_lookup = nn.Parameter(torch.randn(self._vocab_size-1, self._embedding_size))

        self.C_lookup = []
        for hopn in range(self._hops):
            temp = nn.Parameter(torch.randn(self._vocab_size-1, self._embedding_size))
            self.register_parameter("C_{}_lookup".format(hopn+1), temp)
            self.C_lookup.append(temp)

            
    # stories (batch_size, memory_size, sentence_size), e.g., (32, 10, 7)
    # queries (batch_size, sentence_size), e.g., (32, 7)
    # Instead of using nn.Embedding, this function fetches embedding directly.
    # It needs one-hot inputs.
    def _inference(self, stories, queries):
        # (32, 7, 20), 20 is embedding_size
        A_1 = torch.cat((self._nil_word_slot, self.A_1_lookup),0)
        q_emb = torch.matmul(queries, A_1)
        # (32, 7, 20), self._encoding is (7, 20)
        u_0 = q_emb * self._encoding
        # (32, 20)
        u_0 = torch.sum(u_0, 1)

        u = [u_0]
        
        for hopn in range(self._hops):
            
            if hopn == 0:
                m_emb_A = torch.matmul(stories, A_1)
                
            else:
                C_hopn_1 = torch.cat((self._nil_word_slot, self.C_lookup[hopn - 1]))
                m_emb_A = torch.matmul(stories, C_hopn_1)

            # (32, 10, 20)
            m_A = torch.sum(m_emb_A * self._encoding, 2)
            # (32, 1, 20)
            u_temp = torch.unsqueeze(u[-1], 1)

            # (32, 10)
            dotted = torch.sum(m_A * u_temp, 2)
            # (32, 10)
            probs = F.softmax(dotted, 1)
            
            # (32, 10, 1)
            probs_temp = torch.unsqueeze(probs, 2)
            # (32, 10, 7, 20)
            C_hopn = torch.cat((self._nil_word_slot, self.C_lookup[hopn]))
            m_emb_C = torch.matmul(stories, C_hopn)
            # (32, 10, 20)
            m_C = torch.sum(m_emb_C * self._encoding, 2)
            # (32, 20)
            o_k = torch.sum(m_C * probs_temp, 1)
            # (32, 20)
            u_k = u[-1] + o_k
            
            if self._nonlin:
                u_k = self._nonlin(u_k)

            u.append(u_k)
            
        W = self.C_lookup[-1]
        # (32, vocab_size)
        output = torch.matmul(u[-1], W.t())
                
        return output
    
    def batch_fit(self, stories, queries, answers, criterion):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Variable (batch_size, memory_size, sentence_size)
            queries: Variable (batch_size, sentence_size)
            answers: Variable (batch_size, vocab_size)

        Returns:
            loss: Variable (1)
        """
        
        output = self._inference(stories, queries)
        loss = criterion(output, answers)

        return loss    
            
    def predict(self, stories, queries):
        """Predicts answers as targets. For example, [2,3,5,6,0,1...]

        Args:
            stories: Variable (batch_size, memory_size, sentence_size)
            queries: Variable (batch_size, sentence_size)

        Returns:
            answers: Variable (batch_size)
        """

        output = self._inference(stories, queries)
        _, targets = torch.max(output, 1)
        return targets   

        