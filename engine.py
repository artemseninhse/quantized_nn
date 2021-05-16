import models

from argparse import ArgumentParser


class ModelEvaluator:
    
    def __init__(self,
                 model_name,
                 quantization=None,
                 n_bits=8):
        """
        :param model_name: string from ['SimpleTextCNN', 'BERTClassifier']
        :param quantization: string from ['static', 'dynamic', 'aware'] or None
        :param n_bits: integer specyfying the intensity of quantization
        """
        self.model_name = model_name
        self.quantization = {}
        if quantization:
            self.quantization["type"] = quantization
            self.quantization["n_bits"] = n_bits
        self.model = getattr(models, self.model_name)(quantization=self.quantization)
        self.data = ()
        
    def get_data(self):
        """
        Prepare train, validation and test samples for model training and 
        metrics evaluation and return several items for inference examples.
        Samples are taken from IMDB Dataset
        """
        # todo: put data in self.data
        pass
    
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
        self.get_data()
        self.train()
        self.validate()


parser = ArgumentParser()
parser.add_argument("model_name",
                    required=False,
                    default="SimpleTextCNN")
parser.add_argument("quantization",
                    required=False,
                    default="")
parser.add_argument("n_bits",
                    required=False,
                    default="8")


if __name__ == "__main__":
    args = parser.parse_args()
    evaluator = ModelEvaluator(args.model_name,
                               args.quantization,
                               args.n_bits)
    evaluator.evaluate()