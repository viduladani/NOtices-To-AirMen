import os
import torch
import numpy as np
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

class BertEmbeddingsExtractor:
    def __init__(self, model_path):
        """
        Initializes the BertEmbeddingsExtractor.

        Args:
            model_path (str): Path to the pre-trained BERT model.  This should contain the config, vocab, and checkpoint files.
        """
        self.model_path = model_path
        # Load the pre-trained BERT model for masked language modeling, specifying the checkpoint.
        self.model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-13800"), output_hidden_states=True)
        # Load the tokenizer for the pre-trained BERT model.
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    def get_bert_concate_embeddings(self, text, sentence_id):
        """
        Extracts BERT embeddings for a given text.

        Args:
            text (str): The input text to extract embeddings from.
            sentence_id (int): The ID of the sentence (used for creating unique labels).

        Returns:
            tuple: A tuple containing:
                - token_embed_return (list): A list of numpy arrays, each representing the concatenated BERT embeddings for a token.
                - label_id_return (list): A list of unique labels for each token.
                - label_return (list): A list of the original tokens.
        """
        # Add special tokens [CLS] and [SEP] to mark the beginning and end of the text.
        marked_text = "[CLS] " + text +" [SEP]"
        # Tokenize the text using the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)
        # Limit the tokenized text to a maximum length of 62 tokens to prevent exceeding the maximum sequence length (64) after adding [CLS] and [SEP]
        if len(tokenized_text) > 64 - 2:  # Accounting for [CLS] and [SEP] tokens
            tokenized_text = tokenized_text[:64- 2]
        # Convert the tokens to their corresponding IDs in the BERT vocabulary.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        text_index = []

        c=0
        # Print each token and its index
        for tup in zip(tokenized_text, indexed_tokens):
            print('{:<12} {:>6,}'.format(tup[0], tup[1]))
            # Store the token and its index in a list
            text_index.append([tup[0], c])
            c=c+1

        # Create segment IDs (all 1s in this case, indicating a single segment).
        segments_ids = [1] * len(tokenized_text)
        # Convert the token IDs and segment IDs to PyTorch tensors.
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Set the model to evaluation mode (disables dropout, etc.).
        self.model.eval()

        # Disable gradient calculation for inference.
        with torch.no_grad():
            # Pass the token and segment tensors through the BERT model.
            outputs = self.model(tokens_tensor, segments_tensors)
            # Get the hidden states from the output.
            hidden_states = outputs[1]

        # Stack the hidden states to create a tensor of shape (num_layers, batch_size, num_tokens, hidden_size).
        token_embeddings = torch.stack(hidden_states, dim=0)
        # Remove the batch size dimension (which is 1 in this case).
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Permute the dimensions to get a tensor of shape (num_tokens, num_layers, hidden_size).
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_embed = []
        label_id = []
        token_embed_return = []
        label_id_return = []
        label_return = []

        # Iterate over each token and its embeddings.
        for i, token in enumerate(token_embeddings):
            # Concatenate the embeddings from the last two layers.
            cat_vec = torch.cat((token[-1], token[-2]), dim=0)
            # Append the concatenated vector to the list of token embeddings.
            token_embed.append(cat_vec)
            # Create a unique label ID for the token.
            label_id.append(f"{sentence_id}_{text_index[i][1]}_{text_index[i][0]}")
            # Append the original token to the list of labels.
            label_return.append(text_index[i][0])

        # Filter out special tokens ([CLS], [SEP], [UNK]).
        for ind, j in enumerate(text_index):
            if j[0] not in ['[CLS]', '[SEP]', '[UNK]']:
                # Convert the token embedding to a NumPy array and append it to the list of token embeddings to return.
                token_embed_return.append(token_embed[ind].numpy())
                # Append the label ID to the list of label IDs to return.
                label_id_return.append(label_id[ind])
                # Append the original token to the list of labels to return.
                label_return.append(label_return[ind])

        # Return the lists of token embeddings, label IDs, and original tokens.
        return token_embed_return, label_id_return, label_return
