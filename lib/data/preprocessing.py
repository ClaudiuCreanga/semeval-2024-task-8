import re
from cleantext import clean
from lib.utils.constants import PreprocessTextLevel


def text_cleanup(text: str) -> str:
    # TODO: Removed this as the organizers said this is the right way
    # to get the boundry between the human text and the machine text

    # text = text.replace("\n", " ")
    # text = text.replace("\t", " ")
    # text = text.replace("\r", " ")

    return text


def split_text_into_words(text: str) -> [str]:
    text = text_cleanup(text)

    words = [w for w in text.split(" ") if w != ""]

    return words


def _tokenization_norm(text: str) -> str:
    text = text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return text


def _rm_line_break(text: str) -> str:
    text = text.replace("\n", "\\n")
    text = re.sub(r'(?:\\n)*\\n', r'\\n', text)
    text = re.sub(r'^.{0,3}\\n', '', text)
    text = text.replace("\\n", " ")
    return text


def _light_clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def _heavy_clean_text(text: str) -> str:
    text = _rm_line_break(text)

    # remove PLM special tokens
    plm_special_tokens = r'(\<pad\>)|(\<s\>)|(\<\/s\>)|(\<unk\>)|(\<\|endoftext\|\>)'
    text = re.sub(plm_special_tokens, "", text)

    # normalize puncuations
    # moses_norm = MosesPunctNormalizer()
    # text = moses_norm.normalize(text)

    # normalize tokenization
    text = _tokenization_norm(text)

    # remove specific text patterns, e.g,, url, email and phone number
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_punct="",
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="<NUMBER>",
        replace_with_digit="<DIGIT>",
        replace_with_currency_symbol="<CUR>",
        lang="en",
    )

    # keep common puncts only
    punct_pattern = r'[^ A-Za-z0-9.?!,:;\-\[\]\{\}\(\)\'\"]'
    text = re.sub(punct_pattern, '', text)
    # remove specific patterns
    spe_pattern = r'[-\[\]\{\}\(\)\'\"]{2,}'
    text = re.sub(spe_pattern, '', text)
    # remove redundate spaces
    text = " ".join(text.split())

    return text


def preprocess(text: str, preprocess_text_level: PreprocessTextLevel) -> str:
    if preprocess_text_level == PreprocessTextLevel.NONE:
        return text
    elif preprocess_text_level == PreprocessTextLevel.LIGHT:
        return _light_clean_text(text)
    elif preprocess_text_level == PreprocessTextLevel.HEAVY:
        return _heavy_clean_text(text)
    else:
        raise NotImplementedError(
            f"No such preprocess text level {preprocess_text_level}!"
        )
