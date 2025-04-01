import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2

from pathlib import Path

from main import get_predictions
from utils import save_to_json


class ImageAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Demo")
        self.root.geometry("1200x600")

        self.original_image = None
        self.annotated_image = None
        self.results = None

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.create_widgets()

    def create_widgets(self):

        # Original Image
        self.left_frame = ttk.LabelFrame(self.root, text="Original Image")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.original_label = ttk.Label(self.left_frame)
        self.original_label.pack(expand=True, fill="both")

        # Annotated Image
        self.right_frame = ttk.LabelFrame(self.root, text="Annotated Image")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.annotated_label = ttk.Label(self.right_frame)
        self.annotated_label.pack(expand=True, fill="both")

        # Control panel
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.select_btn = ttk.Button(
            self.control_frame, text="Select Image", command=self.load_image
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.download_btn = ttk.Button(
            self.control_frame,
            text="Download JSON",
            state=tk.DISABLED,
            command=self.save_json,
        )
        self.download_btn.pack(side=tk.LEFT, padx=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        cv2_imshow = cv2.imshow
        cv2.imshow = lambda *args, **kwargs: None

        try:
            results, annotated_img = get_predictions(file_path, debug=True)

            self.results = {
                "face_detection": results,
            }

            orig_img = cv2.imread(file_path)

            self.original_image = self.convert_image(orig_img)
            self.annotated_image = self.convert_image(annotated_img)

            self.original_label.configure(image=self.original_image)
            self.annotated_label.configure(image=self.annotated_image)

            self.download_btn.config(state=tk.NORMAL)

        except Exception as e:
            print(f"Error processing image: {e}")
        finally:

            cv2.imshow = cv2_imshow
            cv2.destroyAllWindows()

    def convert_image(self, cv2_image):
        """Convert OpenCV image to Tkinter-compatible format"""
        if cv2_image is None:
            return None

        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        height, width = cv2_image.shape[:2]
        max_size = 600
        if height > max_size or width > max_size:
            ratio = min(max_size / height, max_size / width)
            new_size = (int(width * ratio), int(height * ratio))
            cv2_image = cv2.resize(cv2_image, new_size)

        pil_image = Image.fromarray(cv2_image)
        return ImageTk.PhotoImage(pil_image)

    def save_json(self):
        if not self.results:
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )

        if save_path:
            save_to_json(self.results, Path(save_path))
            print(f"Results saved to {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotationApp(root)
    root.mainloop()
