import torch
import torch.nn as nn
from collections import OrderedDict

class ParseLayer(nn.Module):
    def __init__(self, alpha=50.0):
        super().__init__()

        V, K, Q = self.compute_attention_matrix(alpha)
        self.V = nn.Parameter(V, requires_grad=False)
        self.K = nn.Parameter(K, requires_grad=False)
        self.Q = nn.Parameter(Q, requires_grad=False)

        W1, b1, W2, b2 = self.compute_mlp_matrix()
        self.W1 = nn.Parameter(W1, requires_grad=False)
        self.b1 = nn.Parameter(b1, requires_grad=False)
        self.W2 = nn.Parameter(W2, requires_grad=False)
        self.b2 = nn.Parameter(b2, requires_grad=False)

    def forward(self, input_states):
        # attention
        V, K, Q = self.V, self.K, self.Q

        x_v = torch.matmul(V.unsqueeze(0),
            input_states.unsqueeze(1).unsqueeze(-1)).squeeze()
        x_k = torch.matmul(K.unsqueeze(0),
            input_states.unsqueeze(1).unsqueeze(-1)).squeeze()
        x_q = torch.matmul(Q.unsqueeze(0),
            input_states.unsqueeze(1).unsqueeze(-1)).squeeze()

        x_v = x_v.permute(1, 0, 2)
        x_k = x_k.permute(1, 2, 0)
        x_q = x_q.permute(1, 0, 2)

        q_k = torch.matmul(x_q, x_k)
        q_k = torch.softmax(q_k, dim=-1)
        y_v = torch.matmul(q_k, x_v)

        # output of attention layer
        middle_states = torch.cat(y_v.unbind(), -1)
        # add & (norm)
        middle_states += input_states

        # mlp
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # output of the first hidden layer
        hidden_states = nn.functional.relu(
            torch.matmul(W1.unsqueeze(0), middle_states.unsqueeze(-1)).squeeze() + b1)
        # output of the second layer in mlp
        output_states = torch.matmul(W2.unsqueeze(0), hidden_states.unsqueeze(-1)).squeeze() + b2
        # add & (norm)
        output_states += middle_states

        return output_states

    def compute_attention_matrix(self, alpha):
        V = torch.zeros(12, 64, 768)
        K = torch.zeros(12, 64, 768)
        Q = torch.zeros(12, 64, 768)

        for i in range(0, 8):
            V[i, :, 64*8: 64*9] = torch.diag(torch.ones(64))
            V[i, :, 64*4: 64*5] = torch.diag(-torch.ones(64))

        for i in range(0, 8):
            K[i, :, 64*4: 64*5] = alpha * torch.diag(torch.ones(64))

        for i in range(0, 8):
            Q[i, :, 64*i: 64*(i+1)] = alpha * torch.diag(torch.ones(64))

        return V, K, Q

    def compute_mlp_matrix(self):
        d = 64
        p_emb = torch.cat(
            (torch.zeros(59), torch.ones(1), torch.zeros(4)))
        ones = torch.diag(torch.ones(d))

        W1 = torch.zeros(2304, 768)
        b1 = torch.zeros(2304)
        W2 = torch.zeros(768, 2304)
        b2 = torch.zeros(768)

        # u0 h0
        W1[d*0:d*1, d*0:d*1] = ones
        W1[d*0:d*1, d*4:d*5] = p_emb
        W1[d*0:d*1, d*5:d*6] = -p_emb
        b1[d*0:d*1] = -1.0
        # u0 h1
        W1[d*1:d*2] = W1[d*0:d*1]
        W1[d*1:d*2, d*0:d*1] = -ones
        b1[d*1:d*2] = -1.0
        # u0 h2
        W1[d*2:d*3, d*0:d*1] = -ones
        # u0 h3
        W1[d*3:d*4, d*0:d*1] = ones
        # u0 h4
        W1[d*4:d*5, d*3:d*4] = ones
        W1[d*4:d*5, d*4:d*5] = -p_emb
        # u0 h5
        W1[d*5:d*6, d*3:d*4] = -ones
        W1[d*5:d*6, d*4:d*5] = -p_emb

        # u1 h0
        W1[d*6:d*7, d*1:d*2] = ones
        W1[d*6:d*7, d*4:d*5] = p_emb
        W1[d*6:d*7, d*5:d*6] = -p_emb
        W1[d*6:d*7, d*6:d*7] = -p_emb
        b1[d*6:d*7] = -1.0
        # u1 h1
        W1[d*7:d*8] = W1[d*6:d*7]
        W1[d*7:d*8, d*1:d*2] = -ones
        b1[d*7:d*8] = -1.0
        # u1 h2
        W1[d*8:d*9, d*1:d*2] = -ones
        # u1 h3
        W1[d*9:d*10, d*1:d*2] = ones

        # u2 h0
        W1[d*10:d*11, d*2:d*3] = ones
        W1[d*10:d*11, d*4:d*5] = p_emb
        W1[d*10:d*11, d*5:d*6] = -p_emb
        W1[d*10:d*11, d*6:d*7] = -p_emb
        W1[d*10:d*11, d*7:d*8] = -p_emb
        b1[d*10:d*11] = -1.0
        # u2 h1
        W1[d*11:d*12] = W1[d*10:d*11]
        W1[d*11:d*12, d*2:d*3] = -ones
        b1[d*11:d*12] = -1.0
        # u2 h2
        W1[d*12:d*13, d*2:d*3] = -ones
        # u2 h2
        W1[d*13:d*14, d*2:d*3] = ones

        # f0 h0
        W1[d*14+61, d*4:d*5] = p_emb
        W1[d*14+61, d*5:d*6] = p_emb
        b1[d*14+61] = -1.0

        # f1 h0
        W1[d*15+61, d*4:d*5] = p_emb
        W1[d*15+61, d*5:d*6] = p_emb
        b1[d*15+61] = -1.0
        # f1 h1
        W1[d*16+61, d*4:d*5] = p_emb
        W1[d*16+61, d*5:d*6] = -p_emb
        W1[d*16+61, d*6:d*7] = p_emb
        b1[d*16+61] = -1.0
        # f1 h2
        W1[d*17+61, d*4:d*5] = -p_emb
        b1[d*17+61] = 1.0

        # f2 h0
        W1[d*18+61, d*4:d*5] = p_emb
        W1[d*18+61, d*5:d*6] = p_emb
        b1[d*18+61] = -1.0
        # f2 h1
        W1[d*19+61, d*4:d*5] = p_emb
        W1[d*19+61, d*5:d*6] = -p_emb
        W1[d*19+61, d*6:d*7] = p_emb
        b1[d*19+61] = -1.0
        # f2 h2
        W1[d*20+61, d*4:d*5] = p_emb
        W1[d*20+61, d*5:d*6] = -p_emb
        W1[d*20+61, d*6:d*7] = -p_emb
        W1[d*20+61, d*7:d*8] = p_emb
        b1[d*20+61] = -1.0
        # f2 h3
        W1[d*21+61, d*4:d*5] = -p_emb
        b1[d*21+61] = 1.0

        # f3 h0
        b1[d*22+61] = 1.0

        # g3 h0
        W1[d*23+63, d*4:d*5] = p_emb
        W1[d*23+63, d*5:d*6] = p_emb
        b1[d*23+63] = -1.0

        # ------------------------------------------- #

        # u0 f0 g0
        W2[d*0:d*1, d*0:d*1] = ones
        W2[d*0:d*1, d*1:d*2] = -ones
        W2[d*0:d*1, d*2:d*3] = ones
        W2[d*0:d*1, d*3:d*4] = -ones
        W2[d*0:d*1, d*4:d*5] = ones
        W2[d*0:d*1, d*5:d*6] = -ones
        W2[d*0:d*1, d*14:d*15] = ones

        # u1 f1 g1
        W2[d*1:d*2, d*6:d*7] = ones
        W2[d*1:d*2, d*7:d*8] = -ones
        W2[d*1:d*2, d*8:d*9] = ones
        W2[d*1:d*2, d*9:d*10] = -ones
        W2[d*1:d*2, d*15:d*16] = ones
        W2[d*1:d*2, d*16:d*17] = ones
        W2[d*1:d*2, d*17:d*18] = ones

        # u2 f2 g2
        W2[d*2:d*3, d*10:d*11] = ones
        W2[d*2:d*3, d*11:d*12] = -ones
        W2[d*2:d*3, d*12:d*13] = ones
        W2[d*2:d*3, d*13:d*14] = -ones
        W2[d*2:d*3, d*18:d*19] = ones
        W2[d*2:d*3, d*19:d*20] = ones
        W2[d*2:d*3, d*20:d*21] = ones
        W2[d*2:d*3, d*21:d*22] = ones

        # u3 f3 g3
        W2[d*3:d*4, d*22:d*23] = ones
        W2[d*3:d*4, d*23:d*24] = ones

        return W1, b1, W2, b2


class ReasonLayer(nn.Module):
    def __init__(self, alpha=50.0):
        super().__init__()

        Q, K, V, b_Q, b_K, b_V = self.compute_attention_matrix(alpha)
        self.Q = nn.Parameter(Q, requires_grad=False)
        self.K = nn.Parameter(K, requires_grad=False)
        self.V = nn.Parameter(V, requires_grad=False)
        self.b_Q = nn.Parameter(b_Q, requires_grad=False)
        self.b_K = nn.Parameter(b_K, requires_grad=False)
        self.b_V = nn.Parameter(b_V, requires_grad=False)

        W1, b1, W2, b2 = self.compute_mlp_matrix()
        self.W1 = nn.Parameter(W1, requires_grad=False)
        self.b1 = nn.Parameter(b1, requires_grad=False)
        self.W2 = nn.Parameter(W2, requires_grad=False)
        self.b2 = nn.Parameter(b2, requires_grad=False)

    def forward(self, input_states):
        Q, K, V = self.Q, self.K, self.V
        b_Q, b_K, b_V = self.b_Q, self.b_K, self.b_V

        Q_matrix = (torch.matmul(input_states, Q.permute(1, 0, 2).reshape(768, -1)) + b_Q).reshape(-1, 12, 64).permute(1, 0 ,2) # 12 * L * 64
        K_matrix = (torch.matmul(input_states, K.permute(1, 0, 2).reshape(768, -1)) + b_K).reshape(-1, 12, 64).permute(1, 2 ,0) # 12 * 64 * L
        V_matrix = (torch.matmul(input_states, V.permute(1, 0, 2).reshape(768, -1)) + b_V).reshape(-1, 12, 64).permute(1, 0 ,2) # 12 * L * 64

        attention_weights = nn.functional.softmax(torch.bmm(Q_matrix, K_matrix), dim=-1) # 12 * L * L
        # output of attention layer
        middle_states = torch.bmm(attention_weights, V_matrix).permute(1, 0, 2).reshape(-1, 768)
        # add & (norm)
        middle_states += input_states


        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # output of the first hidden layer
        hidden_states = nn.functional.relu(
            torch.matmul(middle_states, W1) + b1)
        # output of the second layer in mlp
        output_states = torch.matmul(hidden_states, W2) + b2
        output_states += middle_states

        return output_states
        
    def compute_attention_matrix(self, alpha):
        LABEL_A = 63
        LABEL_B = 127
        LABEL_C = 191
        LABEL_D = 255
        BLOCK_SIZE = 64

        Q = torch.zeros((12, 768, 64))
        K = torch.zeros((12, 768, 64))
        V = torch.zeros((12, 768, 64))

        b_Q = torch.zeros(768)
        b_K = torch.zeros(768)
        b_V = torch.zeros(768)

        b_Q[LABEL_A - 1] = 0.25
        b_Q[LABEL_B - 1] = 0.25
        b_Q[LABEL_C - 1] = 0.25

        for i in range(62):
            Q[0][0 * BLOCK_SIZE + i][i] = 1.0
            Q[1][1 * BLOCK_SIZE + i][i] = 1.0
            Q[2][2 * BLOCK_SIZE + i][i] = 1.0

            K[0][3 * BLOCK_SIZE + i][i] = 1.0
            K[1][3 * BLOCK_SIZE + i][i] = 1.0
            K[2][3 * BLOCK_SIZE + i][i] = 1.0

            K[0][LABEL_D][LABEL_A - 1] = 1.0
            K[1][LABEL_D][LABEL_A - 1] = 1.0
            K[2][LABEL_D][LABEL_A - 1] = 1.0

        K = alpha * K

        V[0][LABEL_D][LABEL_A] = 1.0
        V[1][LABEL_D][LABEL_A] = 1.0
        V[2][LABEL_D][LABEL_A] = 1.0

        return Q, K, V, b_Q, b_K, b_V

    def compute_mlp_matrix(self):
        LABEL_A = 63
        LABEL_B = 127
        LABEL_C = 191
        LABEL_D = 255
        BLOCK_SIZE = 64

        W1 = torch.zeros((768, 768))
        b1 = torch.zeros(768)
        W2 = torch.zeros((768, 768))
        b2 = torch.zeros(768)

        W1[LABEL_A][LABEL_A] = 1.0
        W1[LABEL_B][LABEL_B] = 1.0
        W1[LABEL_C][LABEL_C] = 1.0
        W1[LABEL_D][LABEL_D] = 1.0

        W1[LABEL_A][LABEL_D - 2] = 1.0 / 3.0
        W1[LABEL_B][LABEL_D - 2] = 1.0 / 3.0
        W1[LABEL_C][LABEL_D - 2] = 1.0 / 3.0
        b1[LABEL_D - 2] = -0.8

        W1[LABEL_A][LABEL_D - 1] = 1.0 / 3.0
        W1[LABEL_B][LABEL_D - 1] = 1.0 / 3.0
        W1[LABEL_C][LABEL_D - 1] = 1.0 / 3.0
        b1[LABEL_D - 1] = -0.9

        W2[LABEL_A][LABEL_A] = -1.0
        W2[LABEL_B][LABEL_B] = -1.0
        W2[LABEL_C][LABEL_C] = -1.0
        W2[LABEL_D][LABEL_D] = -1.0

        W2[LABEL_D - 2][LABEL_D] = 10.0
        W2[LABEL_D - 1][LABEL_D] = -10.0

        return W1, b1, W2, b2


class LogicBERT(nn.Module):
    def __init__(self):
        super().__init__()

        self.parse_layer = ParseLayer()
        reason_layers = [(str(i), ReasonLayer()) for i in range(11)]
        self.reason_layers = nn.Sequential(OrderedDict(reason_layers))

    def forward(self, input_states):
        return self.reason_layers(self.parse_layer(input_states))