import torch
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn.functional as func
from MLM import MaskedModeling


class ProbRank:
    """
        This class is used to calculate the probabilities and ranks for masked words

        The file was created on     Sat June 15th 2024
            it was last edited on   Tue Sep   3rd 2024

        @author: Miriam S.
    """
    def __init__(self, filename, model, targets):
        """
        initialize and store the relevant variables
        :param filename: filename to corresponding file containing sentences (per line)
                         with exactly one masked token per line where the correct target word is inserted by the code
        :param model: the corresponding BERT model for tokenization and MLM
        :param targets: list of target words (or tokens) in the same order as the sentences in the file
        """
        self.model = model
        self.target_words = targets
        # read in file, relying on main model for MLM
        self.data = MaskedModeling(filename, model)
        # load pretrained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model)
        self.mlm_model = BertForMaskedLM.from_pretrained(self.model, output_attentions=True)

        # zip masked input with corresponding tokens to assign correct token to corresponding sentence with [MASK]
        self.sent_tok_list = list(zip(self.data.data, self.target_words))

    def calculate_probability(self):
        """
        calculate the probability of the given words (or tokens) in the pretrained BERT model
        this function only works with one masked position
            it is intended that words with multiple tokens are fed in as single tokens
            i.e., ["deception", "##s"] would be paired with the sentences:
                "I had some very bad [MASK] ##s lately..."
                "I had some very bad deception [MASK] lately..."
        :return: a dictionary containing ranks together with their target words
        """
        prob_dict = dict()
        # tokenize the input, return pytorch tensors
        for item in self.sent_tok_list:
            print(item)
            # access the masked logits
            mask_logits = self.calculate_logits(item[0])
            # calculate probability of the target word
            target_token_id = self.tokenizer.convert_tokens_to_ids(item[1])
            target_word_prob = mask_logits[target_token_id].item()
            # store the probability with 10 decimals (since the words have very low probabilities)
            # NOTE: the probabilities are stored with the sentence instead of the target token
            #       this happens since tokens can overlap (e.g., ##s) overwriting their values in the dictionary
            prob_dict[item[0]] = '{0:.10f}'.format(target_word_prob)
        return prob_dict

    def calculate_rank(self):
        """
        calculate the rank of the given words in the pretrained BERT model
        this method can only calculate ranks for non-tokenized words
        :return: a dictionary containing target words together with their ranks
        """
        rank_dict = dict()
        for item in self.sent_tok_list:
            # access the masked logits and get target token ID
            mask_logits = self.calculate_logits(item[0])
            target_token_id = self.tokenizer.convert_tokens_to_ids(item[1])

            # rank targets by their probabilities (descending) and get rank (+1, since it starts at 0)
            sorted_probs, sorted_ind = torch.sort(mask_logits, descending=True)
            word_rank = (sorted_ind == target_token_id).nonzero(as_tuple=True)[0].item() + 1
            rank_dict[item[1]] = word_rank

        return rank_dict

    def calculate_logits(self, sentence):
        """
        helper method to calculate logits
        :param sentence: the current sentence used for calculation (first item of self.sent_tok_list)
        :return: the masked logits
        """
        # tokenize the input sentence (pytorch tensors)
        tokenized_input = self.tokenizer.encode(sentence, return_tensors='pt')
        # access index of masked token and generate attention mask relying on tokenized input's shape
        index_mask = (tokenized_input == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
        attention_mask = torch.ones(tokenized_input.shape)  # tensor of all ones, no padding, all tokens attended

        # get output logits from the model
        with torch.no_grad():
            output = self.mlm_model(tokenized_input, attention_mask=attention_mask)
            output_logits = output.logits

        # use softmax to get probabilities and calculate logits
        softmax_logits = func.softmax(output_logits, dim=-1)
        mask_logits = softmax_logits[0, index_mask, :]

        return mask_logits

