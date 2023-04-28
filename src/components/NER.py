from pdfminer.high_level import extract_pages, extract_text
import numpy as np
import sys
from typing import List, Dict, Union
from transformers import (TokenClassificationPipeline, AutoModelForTokenClassification, AutoTokenizer)
from transformers.pipelines import AggregationStrategy
from transformers import pipeline

from src.exception import CustomException
from src.logger import logging


def extract_pdf(file_path):
    """
    Extracts text from a pdf file at the given file path and replaces new lines with spaces.

    Args:
    - file_path: A string representing the path of the pdf file to extract text from.

    Returns:
    - A string representing the extracted text from the pdf file with new lines replaced with spaces.
    """
    try : 
        text = extract_text(file_path)
    except Exception as e :
        CustomException(e,sys)
        
    text_replace = text.replace('\n', ' ')
    
    logging.info('Extracts text from a pdf file at the given file path and replaces new lines with spaces.')

    return text_replace




# from transformers import AutoTokenizer, AutoModelForTokenClassification

# tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
# model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")


##### Process text sample (from wikipedia)


def ner(text_replace, model, tokenizer):
    """
    Uses a pre-trained named entity recognition (NER) model to extract unique keyphrases from the text.
    
    Args:
        text_replace (str): The text to extract keyphrases from.
        model: The pre-trained NER model to use.
        tokenizer: The tokenizer to use with the NER model.
    
    Returns:
        list: A list of unique keyphrases extracted by the pipeline.
    """
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    logging.info('A list of unique keyphrases extracted by the pipeline from Jean-Baptiste/roberta-large-ner-english')
    
    return nlp(text_replace)




# Define keyphrase extraction pipeline


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    """
    A pipeline for keyphrase extraction using token classification.

    Args:
        model (str): The name of the pretrained model to use.
        *args: Additional positional arguments to pass to the parent class.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(self, model: str, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs: List[Dict[str, Union[int, str]]]) -> List[str]:
        """
        Postprocess the outputs of the pipeline.

        Args:
            all_outputs (List[Dict[str, Union[int, str]]]): A list of dictionaries containing the outputs of the pipeline.

        Returns:
            List[str]: A list of unique keyphrases extracted by the pipeline.
        """
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        logging.info('A list of unique keyphrases extracted by the pipeline from ml6team/keyphrase-extraction-distilbert-inspec')
        return np.unique([result.get("word").strip() for result in results])



# class KeyphraseExtractionPipeline(TokenClassificationPipeline):
#     def __init__(self, model, *args, **kwargs):
#         super().__init__(
#             model=AutoModelForTokenClassification.from_pretrained(model),
#             tokenizer=AutoTokenizer.from_pretrained(model),
#             *args,
#             **kwargs
#         )

#     def postprocess(self, all_outputs):
#         results = super().postprocess(
#             all_outputs=all_outputs,
#             aggregation_strategy=AggregationStrategy.FIRST,
#         )
#         return np.unique([result.get("word").strip() for result in results])
    
    
# # Load pipeline
# model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
# extractor = KeyphraseExtractionPipeline(model=model_name)

# # Inference


# keyphrases = extractor(extract_pdf())

# print(keyphrases)