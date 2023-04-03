import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionHead(nn.Module):
    def __init__(self, input_size, output_size):
      super(PredictionHead, self).__init__()
      self.fc1 = nn.Linear(input_size, input_size)
      self.relu = torch.nn.ReLU()
      self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class random_rows_container(nn.Module):

    def __init__(self, vg_classes, clip_vector_size, relations = 0, relation_classes = 0):
        super().__init__()
        self.vg_classes = vg_classes
        self.random_row = nn.Parameter(torch.zeros(1, clip_vector_size))
        self.no_object_row = nn.Parameter(torch.zeros(1, clip_vector_size))
        self.relations = relations
        if self.relations > 0:
            self.no_relation_row = nn.Parameter(torch.zeros(1, clip_vector_size))
            self.relation_classes = relation_classes

    def forward(self, description_embeddings, mode):
        if mode == "objects":
            num_random_rows = self.vg_classes - description_embeddings.shape[0]
            random_rows = self.random_row
            no_object_rows = self.no_object_row.to(device=description_embeddings.device, non_blocking=True)
            random_rows = random_rows.expand(num_random_rows,-1).to(device=description_embeddings.device, non_blocking=True)
            description_embeddings = torch.cat([description_embeddings,random_rows, no_object_rows])
            return description_embeddings
        else:
            num_random_rows = self.relation_classes - description_embeddings.shape[0]
            random_rows = self.random_row
            no_relation_rows = self.no_relation_row.to(device=description_embeddings.device, non_blocking=True)
            random_rows = random_rows.expand(num_random_rows,-1).to(device=description_embeddings.device, non_blocking=True)
            description_embeddings = torch.cat([description_embeddings,random_rows, no_relation_rows])
            return description_embeddings