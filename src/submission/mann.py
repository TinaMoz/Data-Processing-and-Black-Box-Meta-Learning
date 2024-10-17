import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
      
        batch_size, seq_length, num_classes, img_size = input_images.shape

        # Clone labels to modify and zero out the query set labels
        input_labels = input_labels.clone()
        input_labels[:, -1, :, :] = torch.zeros_like(input_labels[:, -1, :, :])

        # Reshape the input tensors
        input_images = input_images.view(batch_size, seq_length * num_classes, img_size)
        input_labels = input_labels.view(batch_size, seq_length * num_classes, num_classes)

        # Concatenate images and labels along the last dimension
        inputs = torch.cat((input_images, input_labels), dim=-1)

        # Pass through LSTM layers
        outputs, _ = self.layer1(inputs)
        outputs, _ = self.layer2(outputs)

        # Reshape the output to the original shape
        predictions = outputs.view(batch_size, seq_length, num_classes, num_classes)

        return predictions
    
    ### END CODE HERE ###


    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################
        loss = None
         ### START CODE HERE ###
        query_preds = preds[:, -1, :, :]
        query_labels = labels[:, -1, :, :]

        # Reshape for cross-entropy
        query_preds = query_preds.reshape(-1, self.num_classes).contiguous()
        query_labels = query_labels.reshape(-1, self.num_classes).contiguous()

        # Compute cross-entropy loss
        loss = F.cross_entropy(query_preds, query_labels.argmax(dim=1))
        ### END CODE HERE ###

        return loss
