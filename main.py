import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class TEMAnalyser:
    def __init__(self):
        self.image = None
        self.processed_image = None
        self.selected_particles = []
        self.current_selection = []
        self.current_mask = None
        self.all_particles_mask = None
        self.threshold = 128
        self.blur_size = 5

        self.root = tk.Tk()
        self.root.title("TEM Particle Analyzer - by KrsMt")

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.setup_gui()

    def setup_gui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        load_btn = tk.Button(control_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5, pady=5)

        threshold_frame = tk.Frame(control_frame)
        threshold_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT)
        self.threshold_slider = tk.Scale(threshold_frame, from_=0, to=255,
                                       orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_slider.set(128)
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        blur_frame = tk.Frame(control_frame)
        blur_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(blur_frame, text="Blur:").pack(side=tk.LEFT)
        self.blur_slider = tk.Scale(blur_frame, from_=1, to=21,
                                  orient=tk.HORIZONTAL, command=self.update_blur)
        self.blur_slider.set(5)
        self.blur_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        selection_frame = tk.Frame(control_frame)
        selection_frame.pack(side=tk.LEFT, padx=5)

        self.process_btn = tk.Button(selection_frame, text="Start Selection",
                                   command=self.start_particle_selection)
        self.process_btn.pack(side=tk.TOP, pady=2)

        self.finish_btn = tk.Button(selection_frame, text="Finish Selection",
                                   command=self.finish_all_selection)
        self.finish_btn.pack(side=tk.TOP, pady=2)

        self.count_label = tk.Label(selection_frame, text="Selected: 0")
        self.count_label.pack(side=tk.TOP, pady=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.process_image()

    def process_image(self):
        if self.image is None:
            return
        blurred = cv2.GaussianBlur(self.image,
                                  (self.blur_size*2+1, self.blur_size*2+1), 0)
        _, thresh = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        self.processed_image = thresh
        self.show_processed_image()

    def update_threshold(self, value):
        self.threshold = int(value)
        self.process_image()

    def update_blur(self, value):
        self.blur_size = int(value)
        self.process_image()

    def show_processed_image(self):
        if self.processed_image is None:
            return

        image = Image.fromarray(self.processed_image)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            image_ratio = image.width / image.height
            canvas_ratio = canvas_width / canvas_height

            if canvas_ratio > image_ratio:
                new_height = canvas_height
                new_width = int(canvas_height * image_ratio)
            else:
                new_width = canvas_width
                new_height = int(canvas_width / image_ratio)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image=image)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.photo, anchor="center"
        )

    def start_particle_selection(self):
        if self.processed_image is None:
            return

        self.current_selection = []
        self.current_mask = np.zeros_like(self.processed_image)

        if self.all_particles_mask is None:
            self.all_particles_mask = np.zeros_like(self.processed_image)

        plt.close('all')

        fig = plt.figure(figsize=(12, 8))
        self.ax = plt.gca()
        self.ax.imshow(self.processed_image, cmap='gray')

        if np.any(self.all_particles_mask):
            self.ax.imshow(self.all_particles_mask, alpha=0.5, cmap='gray')

        self.ax.set_title('Click to select particles (Press Enter to save)')

        self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show(block=True)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.all_particles_mask[y, x] > 0:
            messagebox.showinfo("Notice", "This area has already been selected")
            return

        h, w = self.processed_image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(self.processed_image.copy(), mask, (x, y), 255)

        particle_mask = (mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255

        if np.any(np.logical_and(particle_mask, self.all_particles_mask)):
            messagebox.showinfo("Notice", "This particle overlaps with selected area")
            return

        if np.any(particle_mask):
            self.current_selection.append(particle_mask)
            self.current_mask = cv2.add(self.current_mask, particle_mask)

            self.update_display()

    def update_display(self):
        self.ax.clear()
        self.ax.imshow(self.processed_image, cmap='gray')

        if np.any(self.all_particles_mask):
            self.ax.imshow(self.all_particles_mask, alpha=0.5, cmap='gray')

        if np.any(self.current_mask):
            self.ax.imshow(self.current_mask, alpha=0.3, cmap='cool')

        total_count = len(self.selected_particles) + len(self.current_selection)
        self.ax.set_title(f'Current batch: {len(self.current_selection)} particles (Total: {total_count})')
        plt.draw()

    def on_key(self, event):
        if event.key == 'enter':
            try:
                self.selected_particles.extend(self.current_selection)

                if self.all_particles_mask is None or not np.any(self.all_particles_mask):
                    self.all_particles_mask = self.current_mask.copy()
                else:
                    self.all_particles_mask = cv2.add(self.all_particles_mask, self.current_mask)

                plt.disconnect(self.cid)
                plt.close('all')

                self.count_label.config(text=f"Selected: {len(self.selected_particles)}")

                self.root.update()
            except Exception as e:
                print(f"Error in on_key: {e}")
                messagebox.showerror("Error", f"Error saving selection: {str(e)}")

    def finish_all_selection(self):
        if not self.selected_particles:
            messagebox.showinfo("Notice", "No particles selected yet")
            return

        self.calculate_diameters()

    def calculate_diameters(self):
        if not self.selected_particles:
            return

        diameters = []
        areas = []
        for particle in self.selected_particles:
            area = np.sum(particle > 0)
            areas.append(area)
            diameter = 2 * np.sqrt(area / np.pi)
            diameters.append(diameter)

        result_window = tk.Toplevel(self.root)
        result_window.title("Analysis Results")

        frame = tk.Frame(result_window)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(frame, height=20, width=50, yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=text.yview)

        text.insert(tk.END, "Particle Analysis Results:\n\n")
        text.insert(tk.END, f"{'No.':^6}{'Area(px)':^12}{'Diameter(px)':^12}\n")
        text.insert(tk.END, "-" * 30 + "\n")

        for i, (area, diameter) in enumerate(zip(areas, diameters), 1):
            text.insert(tk.END, f"{i:^6}{area:^12.1f}{diameter:^12.2f}\n")

        text.insert(tk.END, "\nStatistics:\n")
        text.insert(tk.END, f"Total count: {len(diameters)}\n")
        text.insert(tk.END, f"Mean diameter: {np.mean(diameters):.2f} ± {np.std(diameters):.2f}\n")
        text.insert(tk.END, f"Min diameter: {np.min(diameters):.2f}\n")
        text.insert(tk.END, f"Max diameter: {np.max(diameters):.2f}\n")

        export_btn = tk.Button(result_window, text="Export to CSV",
                             command=lambda: self.export_results(areas, diameters))
        export_btn.pack(pady=5)

    def export_results(self, areas, diameters):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Analysis Results"
        )

        if file_path:
            with open(file_path, 'w') as f:
                f.write("No.,Area(px),Diameter(px)\n")
                for i, (area, diameter) in enumerate(zip(areas, diameters), 1):
                    f.write(f"{i},{area:.1f},{diameter:.2f}\n")
            messagebox.showinfo("Success", "Results saved to CSV file")

    def run(self):
        self.root.geometry("800x600")
        self.root.bind("<Configure>", lambda e: self.on_window_resize())
        self.root.mainloop()

    def on_window_resize(self):
        if self.processed_image is not None:
            self.show_processed_image()

if __name__ == "__main__":
    analyser = TEMAnalyser()
    analyser.run()
