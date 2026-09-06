import customtkinter as ctk
import threading
import sys
import os
from tkinter import filedialog
from finetuner import Finetuner
import logging

# Redirect stdout/stderr to a widget
class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
    
    def flush(self):
        pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("All-MiniLM-L6-v2 Finetuner")
        self.geometry("900x600")

        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Finetuner", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, text="Instructions", command=self.sidebar_button_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = ctk.CTkButton(self.sidebar_frame, text="Train", command=self.sidebar_button_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        # Main Frames
        self.instructions_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.train_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")

        self.setup_instructions_frame()
        self.setup_train_frame()

        # Default view
        self.select_frame_by_name("Instructions")

    def setup_instructions_frame(self):
        self.instructions_frame.grid_columnconfigure(0, weight=1)
        
        label = ctk.CTkLabel(self.instructions_frame, text="How to Use", font=ctk.CTkFont(size=24, weight="bold"))
        label.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        text = """
1. Prepare your data:
   - Create a CSV file.
   - It MUST have columns: 'sentence1', 'sentence2', 'label'.
   - 'label' should be a float between 0.0 (dissimilar) and 1.0 (similar).

2. Go to the 'Train' tab.

3. Select your CSV file.

4. Choose an output folder for the finetuned model.

5. Adjust Hyperparameters (optional):
   - Epochs: How many times to iterate over the data.
   - Batch Size: Number of samples per step.

6. Click 'Start Finetuning'.
   - The log window will show progress.
   - Please wait until it says 'Training complete'.
        """
        textbox = ctk.CTkTextbox(self.instructions_frame, width=600, height=400)
        textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        textbox.insert("0.0", text)
        textbox.configure(state="disabled")

    def setup_train_frame(self):
        self.train_frame.grid_columnconfigure(1, weight=1)

        # File Selection
        self.file_path = ctk.StringVar()
        btn_file = ctk.CTkButton(self.train_frame, text="Select Dataset (CSV)", command=self.select_file)
        btn_file.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        lbl_file = ctk.CTkLabel(self.train_frame, textvariable=self.file_path)
        lbl_file.grid(row=0, column=1, padx=20, pady=20, sticky="w")

        # Output Folder
        self.output_path = ctk.StringVar(value="model_output")
        btn_out = ctk.CTkButton(self.train_frame, text="Select Output Folder", command=self.select_folder)
        btn_out.grid(row=1, column=0, padx=20, pady=20, sticky="w")
        lbl_out = ctk.CTkLabel(self.train_frame, textvariable=self.output_path)
        lbl_out.grid(row=1, column=1, padx=20, pady=20, sticky="w")

        # Hyperparameters
        lbl_epochs = ctk.CTkLabel(self.train_frame, text="Epochs:")
        lbl_epochs.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.entry_epochs = ctk.CTkEntry(self.train_frame)
        self.entry_epochs.insert(0, "1")
        self.entry_epochs.grid(row=2, column=1, padx=20, pady=10, sticky="w")

        lbl_batch = ctk.CTkLabel(self.train_frame, text="Batch Size:")
        lbl_batch.grid(row=3, column=0, padx=20, pady=10, sticky="w")
        self.entry_batch = ctk.CTkEntry(self.train_frame)
        self.entry_batch.insert(0, "16")
        self.entry_batch.grid(row=3, column=1, padx=20, pady=10, sticky="w")

        # Start Button
        self.btn_start = ctk.CTkButton(self.train_frame, text="Start Finetuning", command=self.start_training_thread, fg_color="green")
        self.btn_start.grid(row=4, column=0, columnspan=2, padx=20, pady=30)

        # Log Output
        self.log_box = ctk.CTkTextbox(self.train_frame, height=200)
        self.log_box.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        self.log_box.configure(state="disabled")

    def select_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.file_path.set(filename)

    def select_folder(self):
        foldername = filedialog.askdirectory()
        if foldername:
            self.output_path.set(foldername)

    def select_frame_by_name(self, name):
        self.sidebar_button_1.configure(fg_color=("gray75", "gray25") if name == "Instructions" else "transparent")
        self.sidebar_button_2.configure(fg_color=("gray75", "gray25") if name == "Train" else "transparent")

        if name == "Instructions":
            self.instructions_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.instructions_frame.grid_forget()
        
        if name == "Train":
            self.train_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.train_frame.grid_forget()

    def sidebar_button_event(self):
        # Determine which button was clicked is tricky without arguments, 
        # but we can just check the text of the button that called this?
        # Easier to just make separate commands or use lambda, but let's keep it simple.
        # Actually, I'll just bind them separately in __init__ to avoid complexity or use a wrapper.
        pass 
    
    # Redefining to fix the event issue
    def sidebar_button_event(self):
        pass

    def start_training_thread(self):
        if not self.file_path.get():
            self.log("Error: No file selected.\n")
            return
        
        self.btn_start.configure(state="disabled")
        thread = threading.Thread(target=self.run_training)
        thread.start()

    def run_training(self):
        try:
            # Redirect stdout
            sys.stdout = TextRedirector(self.log_box, "stdout")
            sys.stderr = TextRedirector(self.log_box, "stderr")
            
            # Setup logging to also print to stdout so it gets captured
            logging.basicConfig(level=logging.INFO, force=True, handlers=[logging.StreamHandler(sys.stdout)])

            epochs = int(self.entry_epochs.get())
            batch_size = int(self.entry_batch.get())
            
            ft = Finetuner(output_path=self.output_path.get())
            ft.load_data(self.file_path.get())
            ft.train(epochs=epochs, batch_size=batch_size)
            
            self.log("\nDone!\n")

        except Exception as e:
            self.log(f"\nError: {e}\n")
        finally:
            # Restore stdout? Maybe not needed for this app session
            self.btn_start.configure(state="normal")

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = App()
    
    # Fix for the sidebar buttons
    app.sidebar_button_1.configure(command=lambda: app.select_frame_by_name("Instructions"))
    app.sidebar_button_2.configure(command=lambda: app.select_frame_by_name("Train"))
    
    app.mainloop()
