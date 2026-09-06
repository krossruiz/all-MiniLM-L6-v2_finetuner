import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_embeddings(model_path='model_output', data_path='training_data.csv', output_image='embeddings_plot.png', new_sentences=None):
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        print(f"Error: Model path '{model_path}' does not exist. Please train the model first.")
        return

    if not os.path.exists(data_path):
        logger.error(f"Data path not found: {data_path}")
        print(f"Error: Data path '{data_path}' does not exist.")
        return

    logger.info(f"Loading model from {model_path}...")
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Extract unique sentences to visualize
    sentences = list(set(df['sentence1'].tolist() + df['sentence2'].tolist()))
    
    if new_sentences:
        sentences.extend(new_sentences)
        logger.info(f"Added {len(new_sentences)} new sentences.")

    logger.info(f"Found {len(sentences)} unique sentences.")

    logger.info("Generating embeddings...")
    embeddings = model.encode(sentences)

    logger.info("Reducing dimensionality...")
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)
    
    pca_1d = PCA(n_components=1)
    embeddings_1d = pca_1d.fit_transform(embeddings)

    logger.info("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 1D Plot
    y_zeros = [0] * len(embeddings_1d)
    
    num_new = len(new_sentences) if new_sentences else 0
    num_original = len(sentences) - num_new
    
    # Plot regular points
    ax1.scatter(embeddings_1d[:num_original, 0], y_zeros[:num_original], 
                alpha=0.7, c='blue', label='Dataset')
    
    # Plot new sentences
    if new_sentences:
        ax1.scatter(embeddings_1d[num_original:, 0], y_zeros[num_original:], 
                    c='red', s=100, marker='*', label='New Sentences')
        for i in range(num_new):
            idx = num_original + i
            ax1.annotate(sentences[idx], (embeddings_1d[idx, 0], 0), 
                         fontsize=10, fontweight='bold', color='red', rotation=45)

    for i in range(num_original):
        ax1.annotate(sentences[i], (embeddings_1d[i, 0], 0), fontsize=8, alpha=0.75, rotation=45)
        
    ax1.set_title("1D Visualization (PCA)")
    ax1.set_xlabel("PC1")
    ax1.set_yticks([])
    ax1.grid(True, linestyle='--', alpha=0.3)
    if new_sentences: ax1.legend()

    # 2D Plot
    ax2.scatter(embeddings_2d[:num_original, 0], embeddings_2d[:num_original, 1], 
                alpha=0.7, c='green', label='Dataset')

    if new_sentences:
        ax2.scatter(embeddings_2d[num_original:, 0], embeddings_2d[num_original:, 1], 
                    c='red', s=100, marker='*', label='New Sentences')
        for i in range(num_new):
            idx = num_original + i
            ax2.annotate(sentences[idx], (embeddings_2d[idx, 0], embeddings_2d[idx, 1]), 
                         fontsize=10, fontweight='bold', color='red')

    for i in range(num_original):
        ax2.annotate(sentences[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.75)
        
    ax2.set_title("2D Visualization (PCA)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(True, linestyle='--', alpha=0.3)
    if new_sentences: ax2.legend()

    plt.tight_layout()
    plt.savefig(output_image)
    logger.info(f"Plot saved to {output_image}")
    print(f"Visualization saved to {output_image}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize embeddings")
    parser.add_argument("--model_path", type=str, default="model_output", help="Path to the finetuned model")
    parser.add_argument("--data_path", type=str, default="training_data.csv", help="Path to the CSV data")
    parser.add_argument("--output_image", type=str, default="embeddings_plot.png", help="Output image filename")
    parser.add_argument("--new_sentences", type=str, nargs='+', help="New sentences to plot alongside the dataset")
    parser.add_argument("--new_sentences_file", type=str, help="Path to a file containing new sentences (one per line)")
    args = parser.parse_args()

    new_sentences = args.new_sentences if args.new_sentences else []
    if args.new_sentences_file:
        if os.path.exists(args.new_sentences_file):
            with open(args.new_sentences_file, 'r') as f:
                file_sentences = [line.strip() for line in f if line.strip()]
                new_sentences.extend(file_sentences)
        else:
            print(f"Warning: File '{args.new_sentences_file}' not found.")

    visualize_embeddings(model_path=args.model_path, data_path=args.data_path, output_image=args.output_image, new_sentences=new_sentences)

