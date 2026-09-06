from finetuner import Finetuner
import os
import shutil

def test_finetuning():
    print("Starting test...")
    # Clean up previous output
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")

    ft = Finetuner(output_path="test_output")
    ft.load_data("test_data.csv")
    
    # Train for 1 epoch with small batch
    ft.train(epochs=1, batch_size=2)
    
    if os.path.exists("test_output") and os.path.exists(os.path.join("test_output", "config.json")):
        print("SUCCESS: Model output found.")
    else:
        print("FAILURE: Model output not found.")

if __name__ == "__main__":
    test_finetuning()
