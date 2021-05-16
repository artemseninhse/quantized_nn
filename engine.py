import models

from argparse import ArgumentParser
from text_utils import DataConfigurator


class ModelEvaluator:
    
    def __init__(self,
                 model_name="SimpleTextCNN",
                 quantization=None,
                 n_bits=8):
        """
        :param model_name: string from ['SimpleTextCNN', 'BERTClassifier']
        :param quantization: string from ['static', 'dynamic', 'aware'] or None
        :param n_bits: integer specyfying the intensity of quantization
        """
        self.data_configurator = DataConfigurator()
        self.model_name = model_name
        self.quantization = {}
        if quantization:
            self.quantization["type"] = quantization
            self.quantization["n_bits"] = n_bits
        self.data = {}
    
    def train(self):
        """
        Train a model specified
        
        :return: model trained
        """
        # todo: write trainer.py module w train loop, epoch func, configurator etc
        pass
    
    def validate(self):
        """
        Calculate metrics, print inference examples and inference time
        """
        # todo: write validator.py module with metrics, loss, inference time functions, logger of results,
        # function for calculating all main metrics and function for printing examples
        pass
    
    def evaluate(self):
        """
        Train model, calculate test metrics, print inference examples and inference time
        and log metrics, examples and time
        """
        self.data = self.data_configurator.configurate()
        self.model = getattr(models, self.model_name)(vocab_size=len(self.data_configurator.tokenizer.tok2id)+1,
                                                      quantization=self.quantization)
        self.train()
        self.validate()


parser = ArgumentParser()
parser.add_argument("--model_name",
                    required=False,
                    default="SimpleTextCNN")
parser.add_argument("--quantization",
                    required=False,
                    default="")
parser.add_argument("--n_bits",
                    required=False,
                    default="8")


if __name__ == "__main__":
    args = parser.parse_args()
    evaluator = ModelEvaluator(args.model_name,
                               args.quantization,
                               args.n_bits)
    evaluator.evaluate()