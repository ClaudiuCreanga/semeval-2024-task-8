from .bert.bert import BERT
from .bert.bert_bilstm import BERTBiLSTM
from .bert.hierarchical_bert import HierarchicalBERT
from .albert.albert import ALBERT
from .distilbert.distilbert import DistilBERT
from .t5flan.t5flan import T5Flan
from .deberta.deberta import Deberta
from .roberta.roberta import RoBERTa
from .roberta.roberta_bilstm import RoBERTaBiLSTM
from .xlnet.xlnet import XLNet
from .xlm_roberta.xlm_roberta import XLMRoBERTa
from .xlm_roberta.xlm_roberta_bilstm import XLMRoBERTaBiLSTM
from .cnn.statistics_cnn_1d import StatisticsCNN1D
from .gpt2.gpt2 import GPT2
from .longformer.longformer import LongformerForTokenClassification
from .longformer.longformer_crf import LongformerCRFForTokenClassification
from .longformer.longformer_bilstm import LongformerBiLSTMForTokenClassification
from .longformer.longformer_bilstm_crf import LongformerBiLSTMCRFForTokenClassification
from .bilstm.bilstm import BiLSTMForTokenClassification
from .bilstm.bilstm_crf import BiLSTMCRFForTokenClassification
from .bilstm.char_word_bilstm import (
    CharacterAndWordLevelEmbeddingsWithBiLSTMForTokenClassification
)


def get_model(model_name: str, model_config: dict):
    if model_name == "bert":
        return BERT(**model_config)
    elif model_name == "bert_bilstm":
        return BERTBiLSTM(**model_config)
    elif model_name == "hierarchical_bert":
        return HierarchicalBERT(**model_config)
    elif model_name == "albert":
        return ALBERT(**model_config)
    elif model_name == "distilbert":
        return DistilBERT(**model_config)
    elif model_name == "t5flan":
        return T5Flan(**model_config)
    elif model_name == "deberta":
        return Deberta(**model_config)
    elif model_name == "roberta":
        return RoBERTa(**model_config)
    elif model_name == "roberta_bilstm":
        return RoBERTaBiLSTM(**model_config)
    elif model_name == "xlnet":
        return XLNet(**model_config)
    elif model_name == "xlm_roberta":
        return XLMRoBERTa(**model_config)
    elif model_name == "xlm_roberta_bilstm":
        return XLMRoBERTaBiLSTM(**model_config)
    elif model_name == "statistics_cnn_1d":
        return StatisticsCNN1D(**model_config)
    elif model_name == "gpt2":
        return GPT2(**model_config)
    elif model_name == "longformer":
        return LongformerForTokenClassification(**model_config)
    elif model_name == "longformer_crf":
        return LongformerCRFForTokenClassification(**model_config)
    elif model_name == "longformer_bilstm":
        return LongformerBiLSTMForTokenClassification(**model_config)
    elif model_name == "longformer_bilstm_crf":
        return LongformerBiLSTMCRFForTokenClassification(**model_config)
    elif model_name == "bilstm_for_token_classification":
        return BiLSTMForTokenClassification(**model_config)
    elif model_name == "bilstm_crf_for_token_classification":
        return BiLSTMCRFForTokenClassification(**model_config)
    elif model_name == "char_word_bilstm_for_token_classification":
        return CharacterAndWordLevelEmbeddingsWithBiLSTMForTokenClassification(
            **model_config
        )
    else:
        raise NotImplementedError("No such model!")
