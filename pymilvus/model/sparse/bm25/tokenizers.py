import logging
import re
import string
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Match, Optional, Type

import yaml
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_class_registry = {}


def register_class(register_as: str):
    def decorator(cls: Type[Any]):
        _class_registry[register_as] = cls
        return cls

    return decorator


class Preprocessor:
    def apply(self, text: str):
        error_message = "Each preprocessor must implement its 'apply' method."
        raise NotImplementedError(error_message)


@register_class("CharacterfilterPreprocessor")
class CharacterfilterPreprocessor:
    def __init__(self, chars_to_replace: str):
        self.replacement_table = str.maketrans({char: " " for char in chars_to_replace})

    def apply(self, text: str):
        return text.translate(self.replacement_table)


@register_class("ReplacePreprocessor")
class ReplacePreprocessor:
    def __init__(self, replacement_mapping: Dict[str, str]):
        self.replacement_mapping = replacement_mapping
        self.pattern = re.compile("|".join(map(re.escape, replacement_mapping.keys())))

    def _replacement_function(self, match: Match):
        return self.replacement_mapping[match.group(0)]

    def apply(self, text: str):
        return self.pattern.sub(self._replacement_function, text)


@register_class("StandardTokenizer")
class StandardTokenizer:
    def tokenize(self, text: str):
        return word_tokenize(text)


class TextFilter:
    def apply(self, tokens: List[str]):
        error_message = "Each filter must implement the 'apply' method."
        raise NotImplementedError(error_message)


@register_class("LowercaseFilter")
class LowercaseFilter(TextFilter):
    def apply(self, tokens: List[str]):
        return [token.lower() for token in tokens]


@register_class("StopwordFilter")
class StopwordFilter(TextFilter):
    def __init__(self, language: str = "english", stopword_list: Optional[List[str]] = None):
        if stopword_list is None:
            stopword_list = []
        self.stopwords = set(stopwords.words(language) + stopword_list)

    def apply(self, tokens: List[str]):
        return [token for token in tokens if token not in self.stopwords]


@register_class("PunctuationFilter")
class PunctuationFilter(TextFilter):
    def __init__(self, extras: str = ""):
        self.punctuation = set(string.punctuation + extras)

    def apply(self, tokens: List[str]):
        return [token for token in tokens if token not in self.punctuation]


@register_class("StemmingFilter")
class StemmingFilter(TextFilter):
    def __init__(self, language: str = "english"):
        self.stemmer = SnowballStemmer(language)

    def apply(self, tokens: List[str]):
        return [self.stemmer.stem(token) for token in tokens]


class Tokenizer:
    def tokenize(self, text: str):
        error_message = "Each tokenizer must implement its 'tokenize' method."
        raise NotImplementedError(error_message)


@register_class("JiebaTokenizer")
class JiebaTokenizer(Tokenizer):
    def __init__(self):
        if find_spec("jieba") is None:
            error_message = "jieba is required for JiebaTokenizer but is not installed. Please install it using 'pip install jieba'."
            logger.error(error_message)
            raise ImportError(error_message)

    def tokenize(self, text: str):
        import jieba

        return jieba.lcut(text)


@register_class("MecabTokenizer")
class MecabTokenizer(Tokenizer):
    def __init__(self):
        if find_spec("MeCab") is None:
            error_message = "MeCab is required for MecabTokenizer but is not installed. Please install it using 'pip install mecab-python3'."
            logger.error(error_message)
            raise ImportError(error_message)

    def tokenize(self, text: str):
        import MeCab

        wakati = MeCab.Tagger("-Owakati")
        return wakati.parse(text).split()


@register_class("KonlpyTokenizer")
class KonlpyTokenizer(Tokenizer):
    def __init__(self):
        if find_spec("konlpy") is None:
            error_message = "konlpy is required for KonlpyTokenizer but is not installed. Please install it using 'pip install konlpy'."
            logger.error(error_message)
            raise ImportError(error_message)

    def tokenize(self, text: str):
        from konlpy.tag import Kkma

        return Kkma().nouns(text)


class Analyzer:
    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        preprocessors: Optional[List[Preprocessor]] = None,
        filters: Optional[List[TextFilter]] = None,
    ):
        self.name = name
        self.tokenizer = tokenizer
        self.preprocessors = preprocessors
        self.filters = filters

    def __call__(self, text: str):
        for preprocessor in self.preprocessors:
            text = preprocessor.apply(text)
        tokens = self.tokenizer.tokenize(text)
        for _filter in self.filters:
            tokens = _filter.apply(tokens)
        return tokens


def build_default_analyzer(language: str = "en"):
    default_config_path = Path(__file__).parent / "lang.yaml"
    return build_analyer_from_yaml(default_config_path, language)


def build_analyer_from_yaml(filepath: str, name: str):
    with Path(filepath).open() as file:
        config = yaml.safe_load(file)

    lang_config = config.get(name)
    if not lang_config:
        error_message = f"No configuration found {name}"
        raise ValueError(error_message)

    tokenizer_class_type = _class_registry[lang_config["tokenizer"]["class"]]
    tokenizer_params = lang_config["tokenizer"]["params"]

    tokenizer = tokenizer_class_type(**tokenizer_params)
    preprocessors = []
    filters = []
    if "preprocessors" in lang_config:
        preprocessors = [
            _class_registry[filter_config["class"]](**filter_config["params"])
            for filter_config in lang_config["preprocessors"]
        ]
    if "filters" in lang_config:
        filters = [
            _class_registry[filter_config["class"]](**filter_config["params"])
            for filter_config in lang_config["filters"]
        ]

    return Analyzer(name=name, tokenizer=tokenizer, preprocessors=preprocessors, filters=filters)
