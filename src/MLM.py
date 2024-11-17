from transformers import pipeline, BertTokenizer, BertForMaskedLM


def read(filename):
    """
    helper method to read in the file
    :param filename: filename to corresponding file containing sentences (per line)
    :return: a list containing separate lines
    """
    complete_data = []
    # open the file with utf8 encoding, split it at the comma (it should be csv)
    with open(filename, encoding="utf8") as f:
        for line in f:
            complete_data.append(line)

    return complete_data


class MaskedModeling:
    """
        This class is used to perform masked language modeling with one masked token per instance

        The file was created on     Mon May  20th 2024
            it was last edited on   Tue Aug  27th 2024

        @author: Miriam S.
    """
    def __init__(self, filename, model):
        """
        initialize and store the relevant variables
        :param filename: filename to corresponding file containing sentences with exactly one masked token per line
        :param model: the corresponding BERT model for tokenization and MLM
        """
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertForMaskedLM.from_pretrained(model)
        self.data = read(filename)

    def predict_masks(self):
        """
        method to predict words for masked positions
        :return: a list containing top 10 predictions per line in the input file
        """
        line_predictions, top10_predictions = [], []
        # predict then ten most probable masked tokens
        masking = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, top_k=10)
        for line in self.data:
            # save the output to access the prediction probabilities
            result = masking(line)
            for item in result:
                # save each predicted token with its corresponding probability as a tuple
                # originally, 'token_str' contains words with each token separated by whitespace
                # these are removed for readability
                top10_predictions.append((item['token_str'].replace(" ", ""), '{0:.3g}'.format(item['score'])))
            # save all ten predictions per line and reset top 10 list
            line_predictions.append(top10_predictions)
            top10_predictions = []
        return line_predictions

