import logging
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import pandas as pd

class Finetuner:
    def __init__(self, model_name='all-MiniLM-L6-v2', output_path='model_output'):
        self.model_name = model_name
        self.output_path = output_path
        self.model = None
        self.train_examples = []
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        self.logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def load_data(self, file_path):
        self.logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Expecting columns: sentence1, sentence2, label (score 0.0 to 1.0)
        if not all(col in df.columns for col in ['sentence1', 'sentence2', 'label']):
             raise ValueError("CSV must contain 'sentence1', 'sentence2', and 'label' columns.")

        self.train_examples = []
        for index, row in df.iterrows():
            self.train_examples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['label'])))
        
        self.logger.info(f"Loaded {len(self.train_examples)} training examples.")

    def train(self, epochs=1, batch_size=16, warmup_steps=100):
        if not self.model:
            self.load_model()

        if not self.train_examples:
            raise ValueError("No training data loaded.")

        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)

        self.logger.info("Starting training...")
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       epochs=epochs,
                       warmup_steps=warmup_steps,
                       output_path=self.output_path,
                       show_progress_bar=True) # Progress bar prints to stdout, might capture it later
        self.logger.info(f"Training complete. Model saved to {self.output_path}")

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    ft = Finetuner()
    # ft.load_data('test.csv') # Uncomment to test
    # ft.train()
