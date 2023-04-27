from pdfminer.high_level import extract_pages, extract_text
import numpy as np
from transformers import (TokenClassificationPipeline, AutoModelForTokenClassification, AutoTokenizer)
from transformers.pipelines import AggregationStrategy


def extract_pdf(file_path):
    text = extract_text(file_path)
    text_replace = text.replace('\n',' ')
    return text_replace



# from transformers import AutoTokenizer, AutoModelForTokenClassification

# tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
# model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")


##### Process text sample (from wikipedia)

from transformers import pipeline

def ner(text_replace,model,tokenizer):
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    return nlp(text_replace)






# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])
    
# # Load pipeline
# model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
# extractor = KeyphraseExtractionPipeline(model=model_name)

# # Inference


# keyphrases = extractor(extract_pdf())

# print(keyphrases)