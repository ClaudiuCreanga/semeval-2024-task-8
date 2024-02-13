from transformers import (
    PreTrainedTokenizer, BertTokenizer, RobertaTokenizer, XLNetTokenizer,
    XLMRobertaTokenizer, DistilBertTokenizer, AlbertTokenizer, GPT2Tokenizer, LongformerTokenizer, T5TokenizerFast, DebertaV2TokenizerFast, AutoTokenizer
)


def get_tokenizer(model_name: str, pretrained_name: str) -> PreTrainedTokenizer:
    if model_name == "bert":
        return BertTokenizer.from_pretrained(pretrained_name)
    elif model_name == "albert":
        return AlbertTokenizer.from_pretrained(pretrained_name)
    elif model_name == "distilbert":
        return DistilBertTokenizer.from_pretrained(pretrained_name)
    elif model_name == "t5flan":
        return T5TokenizerFast.from_pretrained(pretrained_name)
    elif model_name == "deberta":
        return AutoTokenizer.from_pretrained(pretrained_name)
    elif model_name == "roberta":
        return RobertaTokenizer.from_pretrained(pretrained_name)
    elif model_name == "xlnet":
        return XLNetTokenizer.from_pretrained(pretrained_name)
    elif model_name == "xlm_roberta":
        return XLMRobertaTokenizer.from_pretrained(pretrained_name)
    elif model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif model_name == "longformer":
        return LongformerTokenizer.from_pretrained(pretrained_name)
    else:
        raise NotImplementedError("No such tokenizer!")
