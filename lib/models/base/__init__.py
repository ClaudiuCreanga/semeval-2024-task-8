from abc import ABC, abstractmethod


class BaseModelForTokenClassification(ABC):
    @abstractmethod
    def freeze_transformer_layer(self):
        pass

    @abstractmethod
    def unfreeze_transformer_layer(self):
        pass

    @abstractmethod
    def get_predictions_from_logits(self, logits, labels=None, corresponding_word=None):
        pass
