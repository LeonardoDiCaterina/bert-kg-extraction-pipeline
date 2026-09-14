from .architecture_2 import DynamicKGExtractor
from .architecture_typed_queries import DynamicKGExtractorTypedQueries
from .architecture_disentangled import DynamicKGExtractorDisentangled
from .bipartite_loss import SetCriterion

def build_decoder(decoder_type, **kwargs):
    if decoder_type == "baseline":
        return DynamicKGExtractor(**kwargs)
    elif decoder_type == "typed":
        return DynamicKGExtractorTypedQueries(**kwargs)
    elif decoder_type == "disentangled":
        return DynamicKGExtractorDisentangled(**kwargs)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")
