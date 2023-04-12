from pdfminer.high_level import extract_pages, extract_text
import numpy as np
from transformers import (TokenClassificationPipeline, AutoModelForTokenClassification, AutoTokenizer,)
from transformers.pipelines import AggregationStrategy


def extract_pdf(file_path):
    text = extract_text(file_path)
    text_replace = text.replace('\n',' ')
    return text_replace

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])
    
# # Load pipeline
# model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
# extractor = KeyphraseExtractionPipeline(model=model_name)

# # Inference


# keyphrases = extractor(extract_pdf())

# print(keyphrases)