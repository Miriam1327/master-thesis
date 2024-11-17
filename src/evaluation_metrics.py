import math
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
from scipy import stats
from MLM import MaskedModeling


class EvaluationMetrics:
    """
        This class is used to evaluate model performance with perplexity and surprisal

        The file was created on     Thu May  30th 2024
            it was last edited on   Fri Sep   6th 2024

        @author: Miriam S.
    """
    def __init__(self, filename, model):
        """
        initialize and store the relevant variables
        :param filename: filename to evaluation file containing sentences (per line) for PPL and surprisal calculation
                         NOTE: in order to receive correct predictions, target words are not masked
        :param model: the corresponding BERT model for tokenization and masking
        """
        self.model = model
        self.data = MaskedModeling(filename, model)
        # load pretrained BERT model (weights) and tokenizer (vocab)
        self.tokenizer = BertTokenizer.from_pretrained(self.model)
        self.mlm_model = BertForMaskedLM.from_pretrained(self.model)
        # set model to evaluation mode to make sure it produces reliable and consistent results during inference
        self.mlm_model.eval()

    def calculate_perplexity(self):
        """
        method to calculate perplexity of sentences
            low perplexity indicates more surprising sequence (less confident prediction)
            high perplexity indicates less surprising sequence (more confident prediction)
        perplexity is calculated by taking the exponent of the average negative log-likelihood
        :return: a list containing perplexity scores per sentence in the given file
        """
        total_ppl = []
        for sentence in self.data.data:
            # variables for total log probability and token count
            log_prob, token_count = 0.0, 0

            # tokenize the input sentence (pytorch tensors) and get IDs
            tokenized_input = self.tokenizer.encode(sentence, return_tensors='pt')
            token_ids = self.tokenizer.convert_ids_to_tokens(tokenized_input[0])

            # skip special tokens ([CLS], [SEP]) in iteration
            for i in range(1, len(token_ids) - 1):
                # sequentially mask tokens
                masked_input = tokenized_input.clone()  # clone original input
                masked_input[0, i] = self.tokenizer.mask_token_id

                # get output logits (predictions) from the model
                with torch.no_grad():
                    outputs = self.mlm_model(masked_input)
                    output_logits = outputs.logits

                # use softmax and logits to retrieve probability of corresponding token at currently masked position
                target_id = tokenized_input[0, i].item()
                # softmax will be applied along first dimension - normalization happens across batch for each class
                token_prob = torch.softmax(output_logits[0, i], dim=0)[target_id].item()

                # add up log probabilities and increase token count
                log_prob += np.log(token_prob)
                token_count += 1

            # calculate perplexity with negative log probability and calculated token count
            perplexity = np.exp(-log_prob / token_count)
            total_ppl.append(perplexity)
        return total_ppl

    def calculate_surprisal(self):
        """
        method to calculate surprisal of words in context
            lower surprisal indicates more predictable (less surprising) words
            higher surprisal indicates less predictable (more surprising) words
        surprisal is calculated with the negative logarithm of base b which is 2 in this case
        :return: a list containing tuples of the form (word, surprisal value (in bits))
        """
        surprisals = []
        # iterate over the input data
        for sentence in self.data.data:
            # tokenize the input sentence (pytorch tensors)
            inputs = self.tokenizer(sentence, return_tensors='pt')
            # access the input ids as well as attention masks
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # get output logits (predictions) from the model
            with torch.no_grad():
                outputs = self.mlm_model(input_ids, attention_mask=attention_mask)
                output_logits = outputs.logits

            # use softmax and logits to retrieve probability of corresponding token
            # use dim = -1 for normalization of logits across vocabulary for each position in the sequence
            probs = torch.nn.functional.softmax(output_logits, dim=-1)

            # calculate surprisal for each word in the sentence based on context
            for i, token_id in enumerate(input_ids[0]):
                # skip [CLS] and [SEP] tokens
                if token_id == self.tokenizer.cls_token_id or token_id == self.tokenizer.sep_token_id:
                    continue

                token_prob = probs[0, i, token_id].item()
                surprisal = -math.log2(token_prob)
                # append both the decoded token together with its surprisal value to a list
                # NOTE: the surprisal values are appended for all words in the sentence (not only the target word)
                #       this is the case since target words might be tokenized differently depending on the tokenizer
                surprisals.append((self.tokenizer.decode([token_id]), surprisal))

        return surprisals


def calculate_correlation(value1, value2):
    """
    method to calculate Spearman correlation
    :param value1: list containing values for first category
    :param value2: list containing values for second category
    :return: a tuple containing the correlation coefficient and the assigned p-value
    """
    rho, p_val = stats.spearmanr(value1, value2)

    return rho, p_val
