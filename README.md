# Can BERT Generate Quebec English? 
## Exploring and Improving the Acceptability of Regional Variation in Pretrained Language Models

This repository contains the code for the Master Thesis in Computational Linguistics by Miriam Segiet at the University of Stuttgart.

## Abstract
While pretrained language models have transformed natural language processing by generating contextually rich language representations, their performance can vary widely across different language varieties. Monolingual models, trained on large datasets in a single language, may struggle with language varieties that incorporate elements from multiple languages. This thesis examines the ability of pretrained language models, particularly BERT, to handle Quebec English, a regionally influenced variety of English shaped by contact with French. By comparing three different BERT models—one monolingual, one multilingual, and one fine-tuned on Quebec English-specific data—this study evaluates their effectiveness in generating Quebec English target words and English synonyms within a masked language modeling framework. Results suggest that fine-tuning improves performance for Quebec English-specific target words, outperforming the standard pretrained models. Additionally, findings indicate that tokenization, sentence context, and pretraining data substantially impact prediction accuracy, with all models struggling most with infrequent, region-specific expressions. This work contributes to the broader goal of developing natural language processing tools that inclusively represent diverse linguistic communities, underscoring the importance of fine-tuning in adapting language models to regional and minority language varieties.

The three model versions of BERT are: `bert-base-uncased` and `bert-base-multilingual-uncased` as well as a fine-tuned version of `bert-base-uncased` specifically exposed to Quebec English-specific data. 

This repository contains the code used to investigate BERT's performance on Quebec English in my thesis. The data is taken from previous work by [Miletic et al. (2020)](http://redac.univ-tlse2.fr/corpus/canen/mileticEtAl2020_LREC.pdf) and [Miletic et al. (2023)](http://redac.univ-tlse2.fr/corpus/canen/MileticEtAl2021_EMNLP.pdf) and is not included in this directory but is available [here](http://redac.univ-tlse2.fr/corpus/canen.html).


## Project Structure

- **`src`**: Source code for the project containing all necessary files
  * [MLM.py](src/MLM.py): contains the code for the main masked language modeling task
  * [fine-tuning.py](src/fine-tuning.py): contains the code for fine-tuning the model on specific data
  * [probs_and_ranks.py](src/probs_and_ranks.py): contains the code to calculate both probabilities and ranks for target words
  * [rank_multitoken.py](src/rank_multitoken.py): contains the code to calculate the rank for multitoken words
  * [evaluation_metrics.py](src/evaluation_metrics.py): contains the code with evaluation metrics (perplexity, surprisal, and Spearman correlation)
- **`requirements.txt`**: Necessary depencencies for using the files

## Setup Instructions

### Prerequisites

- **Python Version**: Python 3.9
- **Dependencies**: Dependencies can be found in the [`requirements.txt`](requirements.txt) file

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Miriam1327/master-thesis.git
    cd master-thesis1
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Data
The data for fine-tuning is sourced from the [Montreal Corpus](http://redac.univ-tlse2.fr/corpus/canen.html) as described in [Miletic et al. (2020)](http://redac.univ-tlse2.fr/corpus/canen/mileticEtAl2020_LREC.pdf). The [test set](http://redac.univ-tlse2.fr/misc/canenTestset.html) for evaluation is taken from [Miletic et al. (2023)](http://redac.univ-tlse2.fr/corpus/canen/MileticEtAl2021_EMNLP.pdf). Both the fine-tuning and test data are provided in `.txt` format. Any preliminary adaptations to the data are mentioned in the corresponding Python files.

**Note**: Due to licensing restrictions, the data itself is not included in this repository. You can access the original datasets through the publications and links referenced above.

**Note**: Since data requirements may vary, the preprocessing files are not included in this repository. However, the preprocessing steps applied in this project are described in detail within the thesis.
