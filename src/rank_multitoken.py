from transformers import BertTokenizer, BertForMaskedLM
import torch
import itertools
from MLM import MaskedModeling


class RankMultipleTokens:
    """
        This class is used to perform masked language modeling

        The file was created on     Thu Jun  27th 2024
            it was last edited on   Thu Sep  19th 2024

        @author: Miriam S.
    """
    def __init__(self, model, filename, target_words, percentage, num_token, token_list, batch_size=1000):
        """
        initialize and store the relevant variables
        :param model: the corresponding BERT model for tokenization and MLM
        :param filename: filename to corresponding file containing sentences (per line)
                         in this case the format requires one mask and one token (per line), i.e.,
                         "I had some very bad [MASK] ##s lately..." and
                         "I had some very bad deception [MASK] lately..."
        :param target_words: targets of form [[t1.1, t1.2, ..., t1.N], [t2.1, t2.2, ..., t2.N], [...]]
                             e.g., [['deception', '##s'], [...]] given the above example
        :param percentage: percentage (float) for generating subset of tokens (can be 1 as well, using whole data)
        :param num_token: number of tokens (int) (for given purpose, processing Quebec English data, either 2, 3, or 4)
        :param token_list: list of tokens to be added as special tokens
        :param batch_size: the batch size for processing several batches, default is 1000
        """
        special_tokens = {'additional_special_tokens': token_list}
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.tokenizer.add_special_tokens(special_tokens)  # add special tokens to tokenizer
        self.model = BertForMaskedLM.from_pretrained(model)
        self.model.resize_token_embeddings(len(self.tokenizer))  # update model due to added tokens
        # read in file, relying on main model for MLM
        self.data = MaskedModeling(filename, model)
        self.target_words = target_words
        self.percentage = percentage
        self.num_token = num_token
        self.batch_size = batch_size

        # zip masked input with corresponding tokens
        self.sent_tok_list = list(zip(self.data.data, [tok for items in self.target_words for tok in items]))

        # generate list with initial tokens not starting with ## in the vocabulary
        self.initial_tokens = [token for token, token_id in self.tokenizer.vocab.items()
                               if not token.startswith('##')]
        # generate list with continuation tokens starting with ## in the vocabulary
        self.cont_tokens = [token for token, token_id in self.tokenizer.vocab.items()
                            if token.startswith('##')]

    def get_combinations(self, target):
        """
        helper method to retrieve combinations of tokens for rank estimation
        :param target: a list containing tokens for the target word for which the combinations should be built
        :return: a list of combinations containing tuples (tok, ##tok, ##tok)
        """
        combinations = []
        # different combinations depending on the number of tokens
        if self.num_token == 2:
            # append combinations starting with the first target token, followed by any continuation token
            # and combinations starting with any initial tokens, followed by the second target token
            combinations += list(itertools.product([target[0]], self.cont_tokens))
            combinations += list(itertools.product(self.initial_tokens, [target[1]]))

        elif self.num_token == 3:
            combinations += list(itertools.product([target[0]], self.cont_tokens, self.cont_tokens))
            combinations += list(itertools.product(self.initial_tokens, [target[1]], self.cont_tokens))
            combinations += list(itertools.product(self.initial_tokens, self.cont_tokens, [target[2]]))
        elif self.num_token == 4:
            combinations += list(itertools.product([target[0]], self.cont_tokens, self.cont_tokens,
                                                   self.cont_tokens))
            combinations += list(itertools.product(self.initial_tokens, [target[1]], self.cont_tokens,
                                                   self.cont_tokens))
            combinations += list(itertools.product(self.initial_tokens, self.cont_tokens, [target[2]],
                                                   self.cont_tokens))
            combinations += list(itertools.product(self.initial_tokens, self.cont_tokens,
                                                   self.cont_tokens, [target[3]]))

        return combinations

    def get_combined_rank_efficient(self):
        """
        method to effectively calculate the rank for tokenized words in batches
        :return: a dictionary containing ranks for target words; {target: rank}
        """
        result = dict()
        combined_prob = 1.0

        for i in range(0, len(self.sent_tok_list), self.num_token):
            # get the current chunk of lines (for three tokens the next three lines)
            chunk = self.sent_tok_list[i:i + self.num_token]
            target, mask_indices, probabilities = [], [], []
            for c in chunk:
                target.append(c[1])

                inputs = self.tokenizer.encode(c[0], return_tensors='pt')

                # get mask token index
                mask_token_index = (inputs == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
                mask_indices.append(mask_token_index)

                # get predictions from model
                with torch.no_grad():
                    outputs = self.model(inputs)
                predictions = outputs.logits

                # convert logits to probabilities
                probs = torch.nn.functional.softmax(predictions, dim=-1)
                probabilities.append(probs)

                # get joint probability for all tokens
                prob = self.calculate(inputs, mask_token_index, c[1])
                combined_prob *= prob
            # rank starts at 1
            rank = 1
            # process in batches, calculate joint probabilities
            # increase rank if target probability is smaller than current probability
            for batch in self.create_batches(target):
                if self.num_token == 2:
                    for item1, item2 in batch:  # for double token ['resume', '##s']
                        prob = (
                                probabilities[0][0, mask_indices[0], self.tokenizer.convert_tokens_to_ids(item1)].item() *
                                probabilities[1][0, mask_indices[1], self.tokenizer.convert_tokens_to_ids(item2)].item()
                        )
                        # increase rank if there is a more probable combination available
                        if prob > combined_prob:
                            rank += 1
                if self.num_token == 3:  # for triple token ["rem", "##ark", "##ed"]
                    for item1, item2, item3 in batch:
                        prob = (
                                probabilities[0][0, mask_indices[0], self.tokenizer.convert_tokens_to_ids(item1)].item() *
                                probabilities[1][0, mask_indices[1], self.tokenizer.convert_tokens_to_ids(item2)].item() *
                                probabilities[2][0, mask_indices[2], self.tokenizer.convert_tokens_to_ids(item3)].item()
                        )
                        # increase rank if there is a more probable combination available
                        if prob > combined_prob:
                            rank += 1
                if self.num_token == 4:  # for four tokens ["avail", "##abi", "##lit", "##ies"]
                    for item1, item2, item3, item4 in batch:
                        prob = (
                                probabilities[0][0, mask_indices[0], self.tokenizer.convert_tokens_to_ids(item1)].item() *
                                probabilities[1][0, mask_indices[1], self.tokenizer.convert_tokens_to_ids(item2)].item() *
                                probabilities[2][0, mask_indices[2], self.tokenizer.convert_tokens_to_ids(item3)].item() *
                                probabilities[3][0, mask_indices[3], self.tokenizer.convert_tokens_to_ids(item4)].item()
                        )
                        # increase rank if there is a more probable combination available
                        if prob > combined_prob:
                            rank += 1
            # store joined word (as one token) with rank in dictionary and return it
            result[''.join(target)] = rank
        return result

    def create_batches(self, target):
        """
        helper method to create batches for faster processing
        :param target: a list containing tokens for the target word for which the combinations should be built
        :return: yields batch of combinations
        """
        # generate batches of the combinations for faster processing
        combinations = self.get_combinations(target)
        num_combinations = len(combinations)
        for idx in range(0, num_combinations, self.batch_size):
            yield combinations[idx:min(idx + self.batch_size, num_combinations)]

    def calculate(self, input_sentence, mask_index, token):
        """
        helper method to calculate the probability of the token in context
        :param input_sentence: the input sentence for context
        :param mask_index: the specific index of the masked token
        :param token: the target token
        :return: the probability of the token in context
        """
        with torch.no_grad():
            output = self.model(input_sentence)

        # the logits for the mask position
        predictions = output.logits[0, mask_index]

        # calculate probability for the token
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        probability = probs[token_id].item()
        return probability
