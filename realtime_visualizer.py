import customtkinter as ctk
import tkinter as tk
from tkinter import colorchooser
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pandas as pd
import threading
import os
import argparse

class RealtimeVisualizer(ctk.CTk):
    def __init__(self, model_path, data_path, initial_sentences=None):
        super().__init__()

        self.title("Realtime Embedding Visualizer")
        self.geometry("1200x800")
        self.configure(fg_color="#352879") # C64 Blue

        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.pca_2d = None
        self.pca_1d = None
        self.train_embeddings_2d = None
        self.train_embeddings_1d = None
        
        # List of dicts: {'text': str, 'color': str}
        self.saved_items = []
        if initial_sentences:
            for s in initial_sentences:
                self.saved_items.append({'text': s, 'color': '#1f77b4'}) # Default blue

        self.current_sentence = ""
        self.current_color = "#1f77b4" # Default matplotlib blue-ish
        self.debounce_timer = None

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Panel
        self.left_frame = ctk.CTkFrame(self, width=350, corner_radius=0, fg_color="#352879")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_rowconfigure(4, weight=1)

        self.lbl_input = ctk.CTkLabel(self.left_frame, text="Type a sentence:", font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
        self.lbl_input.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        self.entry_input = ctk.CTkEntry(self.left_frame, placeholder_text="Start typing...")
        self.entry_input.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.entry_input.bind("<KeyRelease>", self.on_key_release)

        # Color Picker Row
        self.color_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.color_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_color = ctk.CTkButton(self.color_frame, text="Pick Color", command=self.pick_color, width=100)
        self.btn_color.pack(side="left", padx=(0, 10))
        
        self.color_preview = ctk.CTkLabel(self.color_frame, text="      ", fg_color=self.current_color, corner_radius=5)
        self.color_preview.pack(side="left")

        self.btn_add = ctk.CTkButton(self.left_frame, text="Add to List", command=self.add_sentence)
        self.btn_add.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Scrollable List for Items
        self.list_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="Saved Sentences", fg_color="#352879", label_text_color="white")
        self.list_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

        # Right Panel
        self.right_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#352879")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)

        # Matplotlib Figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.patch.set_facecolor('#352879') # C64 Blue Figure Background
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Initialize Logic
        self.status_label = ctk.CTkLabel(self.right_frame, text="Loading model...", text_color="orange")
        self.status_label.grid(row=1, column=0, pady=10)
        
        threading.Thread(target=self.load_resources).start()
        self.refresh_list_ui()

    def load_resources(self):
        if not os.path.exists(self.model_path):
            self.status_label.configure(text="Error: Model not found!", text_color="red")
            return

        try:
            self.model = SentenceTransformer(self.model_path)
            
            if os.path.exists(self.data_path):
                df = pd.read_csv(self.data_path)
                sentences = list(set(df['sentence1'].tolist() + df['sentence2'].tolist()))
                embeddings = self.model.encode(sentences)
                
                self.pca_2d = PCA(n_components=2)
                self.train_embeddings_2d = self.pca_2d.fit_transform(embeddings)
                
                self.pca_1d = PCA(n_components=1)
                self.train_embeddings_1d = self.pca_1d.fit_transform(embeddings)
                
                self.status_label.configure(text="Ready", text_color="#588d43") # C64 Green-ish
                self.after(0, lambda: self._draw_plots([], [], [], None, None)) # Initial draw
                self.update_plots() # Draw initial items
            else:
                self.status_label.configure(text="Warning: Training data not found.", text_color="yellow")
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="red")

    def pick_color(self):
        color = colorchooser.askcolor(color=self.current_color)[1]
        if color:
            self.current_color = color
            self.color_preview.configure(fg_color=color)
            self.update_plots()

    def on_key_release(self, event):
        if self.debounce_timer:
            self.debounce_timer.cancel()
        self.debounce_timer = threading.Timer(0.5, self.process_input)
        self.debounce_timer.start()

    def process_input(self):
        self.current_sentence = self.entry_input.get()
        self.update_plots()

    def add_sentence(self):
        text = self.entry_input.get()
        if text:
            self.saved_items.append({'text': text, 'color': self.current_color})
            self.entry_input.delete(0, "end")
            self.current_sentence = ""
            self.refresh_list_ui()
            self.update_plots()

    def edit_item(self, index):
        if not (0 <= index < len(self.saved_items)):
            return
            
        item = self.saved_items[index]
        
        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit Sentence")
        dialog.geometry("400x200")
        dialog.configure(fg_color="#352879")
        
        # Make it modal
        dialog.transient(self)
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="Sentence:", text_color="white").pack(pady=(10, 5))
        entry = ctk.CTkEntry(dialog, width=300)
        entry.pack(pady=5)
        entry.insert(0, item['text'])
        
        color_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        color_frame.pack(pady=10)
        
        current_edit_color = [item['color']] # List to be mutable in inner function
        
        color_preview = ctk.CTkLabel(color_frame, text="      ", fg_color=current_edit_color[0], corner_radius=5)
        color_preview.pack(side="left", padx=10)
        
        def pick_edit_color():
            color = colorchooser.askcolor(color=current_edit_color[0])[1]
            if color:
                current_edit_color[0] = color
                color_preview.configure(fg_color=color)
        
        ctk.CTkButton(color_frame, text="Change Color", command=pick_edit_color).pack(side="left")
        
        def save_edit():
            new_text = entry.get()
            if new_text:
                self.saved_items[index] = {'text': new_text, 'color': current_edit_color[0]}
                self.refresh_list_ui()
                self.update_plots()
                dialog.destroy()
        
        ctk.CTkButton(dialog, text="Save", command=save_edit).pack(pady=10)

    def delete_item(self, index):
        if 0 <= index < len(self.saved_items):
            del self.saved_items[index]
            self.refresh_list_ui()
            self.update_plots()

    def refresh_list_ui(self):
        # Clear existing widgets in scrollable frame
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        for i, item in enumerate(self.saved_items):
            row_frame = ctk.CTkFrame(self.list_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)
            
            # Color indicator
            lbl_color = ctk.CTkLabel(row_frame, text="  ", fg_color=item['color'], width=20, corner_radius=5)
            lbl_color.pack(side="left", padx=5)
            
            # Text
            lbl_text = ctk.CTkLabel(row_frame, text=item['text'], anchor="w", text_color="white")
            lbl_text.pack(side="left", fill="x", expand=True, padx=5)
            
            # Edit button
            btn_edit = ctk.CTkButton(row_frame, text="Edit", width=40, fg_color="gray", 
                                    command=lambda idx=i: self.edit_item(idx))
            btn_edit.pack(side="right", padx=2)

            # Delete button
            btn_del = ctk.CTkButton(row_frame, text="X", width=30, fg_color="red", 
                                    command=lambda idx=i: self.delete_item(idx))
            btn_del.pack(side="right", padx=2)

    def update_plots(self):
        if not self.model or not self.pca_2d:
            return

        threading.Thread(target=self._update_plots_thread).start()

    def _update_plots_thread(self):
        saved_emb_2d = []
        saved_emb_1d = []
        saved_colors = []
        
        if self.saved_items:
            texts = [item['text'] for item in self.saved_items]
            saved_colors = [item['color'] for item in self.saved_items]
            embs = self.model.encode(texts)
            saved_emb_2d = self.pca_2d.transform(embs)
            saved_emb_1d = self.pca_1d.transform(embs)

        curr_emb_2d = None
        curr_emb_1d = None
        if self.current_sentence:
            emb = self.model.encode([self.current_sentence])
            curr_emb_2d = self.pca_2d.transform(emb)
            curr_emb_1d = self.pca_1d.transform(emb)
        
        self.after(0, lambda: self._draw_plots(saved_emb_1d, saved_emb_2d, saved_colors, curr_emb_1d, curr_emb_2d))

    def _draw_plots(self, saved_1d, saved_2d, saved_colors, curr_1d, curr_2d):
        self.ax1.clear()
        self.ax2.clear()
        
        # Apply C64 Styling to Axes
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#352879') # Blue background
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.grid(True, linestyle='--', alpha=0.3, color='white')

        # Background Data
        if self.train_embeddings_1d is not None:
            y_zeros_bg = [0] * len(self.train_embeddings_1d)
            self.ax1.scatter(self.train_embeddings_1d[:, 0], y_zeros_bg, alpha=0.3, c='#888888', label='Training Data', s=10)
            self.ax2.scatter(self.train_embeddings_2d[:, 0], self.train_embeddings_2d[:, 1], alpha=0.3, c='#888888', label='Training Data', s=10)

        # Saved
        if len(saved_1d) > 0:
            y_zeros_saved = [0] * len(saved_1d)
            self.ax1.scatter(saved_1d[:, 0], y_zeros_saved, c=saved_colors, s=50, label='Saved', edgecolors='white', linewidths=0.5)
            self.ax2.scatter(saved_2d[:, 0], saved_2d[:, 1], c=saved_colors, s=50, label='Saved', edgecolors='white', linewidths=0.5)
            for i, item in enumerate(self.saved_items):
                self.ax1.annotate(item['text'], (saved_1d[i, 0], 0), rotation=45, fontsize=8, alpha=0.8, color='white')
                self.ax2.annotate(item['text'], (saved_2d[i, 0], saved_2d[i, 1]), fontsize=8, alpha=0.8, color='white')

        # Current
        if curr_1d is not None:
            self.ax1.scatter(curr_1d[:, 0], [0], c=self.current_color, s=150, marker='*', label='Current', edgecolors='white')
            self.ax2.scatter(curr_2d[:, 0], curr_2d[:, 1], c=self.current_color, s=150, marker='*', label='Current', edgecolors='white')
            self.ax1.annotate(self.current_sentence, (curr_1d[0, 0], 0), color='white', fontweight='bold', rotation=45)
            self.ax2.annotate(self.current_sentence, (curr_2d[0, 0], curr_2d[0, 1]), color='white', fontweight='bold')

        self.ax1.set_title("1D Visualization")
        self.ax1.set_yticks([])
        
        self.ax2.set_title("2D Visualization")
        
        self.canvas.draw()

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    
    parser = argparse.ArgumentParser(description="Realtime Embedding Visualizer")
    parser.add_argument("--model_path", type=str, default="model_output", help="Path to the finetuned model")
    parser.add_argument("--data_path", type=str, default="training_data.csv", help="Path to the CSV data")
    parser.add_argument("--new_sentences", type=str, nargs='+', help="Initial sentences to populate")
    parser.add_argument("--new_sentences_file", type=str, help="File containing initial sentences")
    args = parser.parse_args()

    initial_sentences = []
    if args.new_sentences:
        initial_sentences.extend(args.new_sentences)
    
    if args.new_sentences_file and os.path.exists(args.new_sentences_file):
        with open(args.new_sentences_file, 'r') as f:
            initial_sentences.extend([line.strip() for line in f if line.strip()])

    app = RealtimeVisualizer(model_path=args.model_path, data_path=args.data_path, initial_sentences=initial_sentences)
    app.mainloop()
