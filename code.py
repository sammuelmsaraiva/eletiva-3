import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


class SelfAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scaling_factor = math.sqrt(self.embed_dim)
        scores = scores / scaling_factor

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 1
    sequence_length = 5
    embedding_dimension = 8

    sample_input = torch.randn(batch_size, sequence_length, embedding_dimension)

    attention_layer = SelfAttention(embedding_dimension)

    attention_output, attention_matrix = attention_layer(sample_input)

    print("Formato da saída:", attention_output.shape)
    print("Formato da matriz de atenção:", attention_matrix.shape)

    plt.figure()
    plt.imshow(attention_matrix[0].detach().numpy())
    plt.colorbar()
    plt.title("Mapa de Calor - Pesos de Atenção")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.savefig('attention_heatmap.png')
    print("Mapa de calor salvo como 'attention_heatmap.png'")