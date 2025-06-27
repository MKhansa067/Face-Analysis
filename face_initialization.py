import os
import math
import cv2
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from deepface import DeepFace

DATA_FILE = "face_data.csv"
IMAGE_FOLDER = "captured_faces"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

AGE_ADJUSTMENT = -3 

BG_COLOR = "#f5f5f5"
PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
TEXT_COLOR = "#333333"

def initialize_data():
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["filename", "gender", "age_range", "emotion", "race"])
df = initialize_data()

def safe_age_conversion(age_value):
    try:
        if age_value is None or (isinstance(age_value, float) and math.isnan(age_value)):
            return None
        return int(age_value)
    except Exception:
        return None

def calculate_age_range(age):
    if age is None:
        return "Unknown"
    adjusted_age = max(age + AGE_ADJUSTMENT, 0)
    base = (adjusted_age // 5) * 5
    return f"{base}-{base + 5}"

class FaceAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.runtime_vars()

    def setup_window(self):
        self.root.title("Face Analysis")
        self.root.geometry("1100x800")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(1000, 700)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TNotebook', background=BG_COLOR)
        style.configure('TNotebook.Tab', 
                       background=BG_COLOR, 
                       foreground=TEXT_COLOR,
                       padding=[10, 5],
                       font=('Helvetica', 10, 'bold'))
        
        style.configure('TFrame', background=BG_COLOR)
        style.configure('TButton', 
                       font=('Helvetica', 9),
                       padding=6,
                       background=SECONDARY_COLOR,
                       foreground='white')
        
        style.map('TButton',
                 background=[('active', PRIMARY_COLOR)],
                 foreground=[('active', 'white')])

    def runtime_vars(self):
        self.camera_active = False
        self.video_capture = None
        self.last_captured_frame = None
        self.current_image_path = None
        self.current_sort_column = None
        self.sort_ascending = True

    def create_widgets(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        self.create_notebook()
        self.create_camera_tab()
        self.create_upload_tab()
        self.create_bottom_section()

    def create_notebook(self):
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill='both', expand=True)

    def create_camera_tab(self):
        self.camera_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_tab, text="Live Analysis")

        camera_container = ttk.Frame(self.camera_tab)
        camera_container.pack(fill='both', expand=True, padx=10, pady=10)

        video_frame = ttk.Frame(camera_container, width=640, height=480)
        video_frame.pack_propagate(False)
        video_frame.pack(pady=10)
        
        self.video_display = tk.Label(video_frame, bg='black')
        self.video_display.pack(fill='both', expand=True)

        button_frame = ttk.Frame(camera_container)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Start Camera", command=self.start_camera).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Capture", command=self.capture_face).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Stop Camera", command=self.stop_camera).grid(row=0, column=2, padx=5)

    def create_upload_tab(self):
        self.upload_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.upload_tab, text="Image Analysis")

        upload_container = ttk.Frame(self.upload_tab)
        upload_container.pack(fill='both', expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(upload_container)
        control_frame.pack(pady=10)
    
        ttk.Button(control_frame, text="Select Image", command=self.select_image).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Analyze", command=self.analyze_image).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Save Data", command=self.save_image_data).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Delete Selected", command=self.delete_selected).grid(row=0, column=3, padx=5)

        display_frame = ttk.Frame(upload_container)
        display_frame.pack(fill='both', expand=True)

        img_display_container = ttk.Frame(display_frame, width=640, height=480)
        img_display_container.pack_propagate(False)
        img_display_container.grid(row=0, column=0, padx=10, pady=10, sticky='nw')
    
        self.image_display = tk.Label(img_display_container, bg='white')
        self.image_display.pack(fill='both', expand=True)

        results_frame = ttk.Frame(display_frame)
        results_frame.grid(row=0, column=1, sticky='nw', padx=10, pady=10)
    
        self.gender_result = tk.StringVar(value="Gender: -")
        self.age_result = tk.StringVar(value="Age Range: -")
        self.emotion_result = tk.StringVar(value="Emotion: -")
        self.race_result = tk.StringVar(value="Race: -")
    
        tk.Label(results_frame, textvariable=self.gender_result, 
            font=('Helvetica', 10), anchor='w').pack(anchor='w', pady=2)
        tk.Label(results_frame, textvariable=self.age_result, 
            font=('Helvetica', 10), anchor='w').pack(anchor='w', pady=2)
        tk.Label(results_frame, textvariable=self.emotion_result, 
            font=('Helvetica', 10), anchor='w').pack(anchor='w', pady=2)
        tk.Label(results_frame, textvariable=self.race_result, 
            font=('Helvetica', 10), anchor='w').pack(anchor='w', pady=2)
    
        self.status_label = tk.Label(upload_container, text="", fg=PRIMARY_COLOR)
        self.status_label.pack()

    def create_bottom_section(self):
        bottom_container = ttk.Frame(self.main_container)
        bottom_container.pack(fill='x', pady=(10, 0))

        search_frame = ttk.Frame(bottom_container)
        search_frame.pack(fill='x', pady=(0, 5))
    
        tk.Label(search_frame, text="Search:").pack(side='left')
    
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side='left', fill='x', expand=True, padx=5)
        self.search_entry.bind("<KeyRelease>", self.search_data)
    
        self.search_category = tk.StringVar(value="all")
        search_options = ["all", "filename", "gender", "age_range", "emotion", "race"]
        search_dropdown = ttk.OptionMenu(search_frame, self.search_category, "all", *search_options)
        search_dropdown.pack(side='left', padx=5)
    
        ttk.Button(search_frame, text="Clear", command=self.clear_search).pack(side='left', padx=5)
    
        table_frame = ttk.Frame(bottom_container)
        table_frame.pack(fill='both', expand=True)
    
        columns = ["filename", "gender", "age_range", "emotion", "race"]
        self.data_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
    
        for col in columns:
            self.data_table.heading(col, text=col.title(), 
                              command=lambda c=col: self.sort_table(c))
            self.data_table.column(col, width=150, anchor='center')
    
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.data_table.yview)
        self.data_table.configure(yscrollcommand=scrollbar.set)
    
        self.data_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
        self.data_table.bind("<<TreeviewSelect>>", self.on_row_select)
        self.update_data_table()

        control_frame = ttk.Frame(bottom_container)
        control_frame.pack(fill='x', pady=(5, 0))
    
        ttk.Button(control_frame, text="Delete Selected", command=self.delete_selected).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Reset Sort", command=self.reset_sort).pack(side='left', padx=5)

    def search_data(self, event=None):
        search_term = self.search_var.get().lower()
        category = self.search_category.get()
        
        if not search_term:
            self.update_data_table()
            return
            
        global df
        if category == "all":
            filtered_df = df[df.apply(lambda row: any(search_term in str(cell).lower() for cell in row), axis=1)]
        else:
            filtered_df = df[df[category].str.lower().str.contains(search_term)]
            
        self.update_data_table(filtered_df)

    def clear_search(self):
        self.search_var.set("")
        self.search_category.set("all")
        self.update_data_table()

    def sort_table(self, column):
        global df
        
        if self.current_sort_column == column:
            self.sort_ascending = not self.sort_ascending
        else:
            self.current_sort_column = column
            self.sort_ascending = True
            
        df = df.sort_values(by=column, ascending=self.sort_ascending)
        self.update_data_table()

    def reset_sort(self):
        global df
        df = initialize_data()
        self.current_sort_column = None
        self.sort_ascending = True
        self.update_data_table()

    def start_camera(self):
        if self.camera_active:
            return
            
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Unable to access camera")
            return
            
        self.camera_active = True
        self.update_video_feed()

    def stop_camera(self):
        self.camera_active = False
        if self.video_capture:
            self.video_capture.release()
        self.video_display.config(image='')

    def update_video_feed(self):
        if not self.camera_active:
            return
            
        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_camera()
            return
            
        processed_frame = self.process_video_frame(frame)
        self.last_captured_frame = processed_frame.copy()
        
        display_image = self.convert_frame_for_display(processed_frame)
        self.video_display.config(image=display_image)
        self.video_display.image = display_image
        self.video_display.after(20, self.update_video_feed)

    def process_video_frame(self, frame):
        try:
            results = DeepFace.analyze(
                frame, 
                actions=['gender', 'emotion', 'age', 'race'], 
                enforce_detection=False
            )
            
            if not isinstance(results, list):
                results = [results]
                
            for result in results:
                face_region = result.get("region", {})
                x, y, w, h = face_region.get("x", 0), face_region.get("y", 0), face_region.get("w", 0), face_region.get("h", 0)
                
                if w == 0 or h == 0:
                    continue
                    
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Get analysis results
                gender = result.get("dominant_gender", "-")
                emotion = result.get("dominant_emotion", "-")
                race = result.get("dominant_race", "-")
                age = safe_age_conversion(result.get("age"))
                age_range = calculate_age_range(age)
                
                # Display results on frame
                cv2.putText(frame, f"{gender}, {age_range}", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y + h + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, race, (x, y + h + 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                          
        except Exception as e:
            print(f"Frame processing error: {e}")
            
        return frame

    def capture_face(self):
        if self.last_captured_frame is None:
            messagebox.showwarning("Warning", "No frame available to capture")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_{timestamp}.jpg"
        filepath = os.path.join(IMAGE_FOLDER, filename)
        
        cv2.imwrite(filepath, self.last_captured_frame)
        self.analyze_and_store(filepath, show_confirmation=True)

    def select_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if filepath:
            self.current_image_path = filepath
            self.display_selected_image(filepath)
            self.clear_analysis_results()

    def display_selected_image(self, filepath):
        try:
            image = Image.open(filepath)
            image.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(image)
            
            self.image_display.config(image=photo)
            self.image_display.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Unable to display image: {e}")

    def analyze_image(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "No image selected")
            return
            
        self.status_label.config(text="Analyzing... Please wait")
        self.root.update()
        
        self.display_analysis_results(self.current_image_path)
        self.status_label.config(text="")

    def save_image_data(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "No image selected")
            return
            
        self.analyze_and_store(self.current_image_path, show_confirmation=True)

    def analyze_and_store(self, image_path, show_confirmation=False):
        global df
        
        try:
            analysis = DeepFace.analyze(
                image_path, 
                actions=['gender', 'emotion', 'age', 'race'], 
                enforce_detection=False
            )[0]
            
            gender = analysis.get("dominant_gender", "-")
            emotion = analysis.get("dominant_emotion", "-")
            race = analysis.get("dominant_race", "-")
            age = safe_age_conversion(analysis.get("age"))
            age_range = calculate_age_range(age)
            
            filename = os.path.basename(image_path)
            destination = os.path.join(IMAGE_FOLDER, filename)
            
            # Save copy if not already in our folder
            if image_path != destination:
                Image.open(image_path).save(destination)
            
            # Add to dataframe
            new_entry = {
                "filename": filename,
                "gender": gender,
                "age_range": age_range,
                "emotion": emotion,
                "race": race
            }
            
            global df
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            
            self.update_data_table()
            
            if show_confirmation:
                messagebox.showinfo("Saved", f"Image and analysis saved as {filename}")
            
            if self.current_image_path == image_path:
                self.display_analysis_results(image_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def display_analysis_results(self, image_path):
        try:
            analysis = DeepFace.analyze(
                image_path, 
                actions=['gender', 'emotion', 'age', 'race'], 
                enforce_detection=False
            )[0]
            
            self.gender_result.set(f"Gender: {analysis.get('dominant_gender', '-')}")
            self.age_result.set(f"Age Range: {calculate_age_range(safe_age_conversion(analysis.get('age')))}")
            self.emotion_result.set(f"Emotion: {analysis.get('dominant_emotion', '-')}")
            self.race_result.set(f"Race: {analysis.get('dominant_race', '-')}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Unable to analyze image: {str(e)}")

    def update_data_table(self, data=None):
        self.data_table.delete(*self.data_table.get_children())
        
        display_data = data if data is not None else df
        
        for _, row in display_data.iterrows():
            self.data_table.insert("", "end", values=list(row))

    def delete_selected(self):
        selected = self.data_table.selection()
        if not selected:
            messagebox.showwarning("Warning", "No entry selected")
            return
            
        filename = self.data_table.item(selected[0], "values")[0]
        
        if not messagebox.askyesno("Confirm", f"Delete data and file for {filename}?"):
            return
            
        global df
        df = df[df["filename"] != filename]
        df.to_csv(DATA_FILE, index=False)
        
        file_path = os.path.join(IMAGE_FOLDER, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
                
        self.update_data_table()
        
        # Clear display if showing the deleted file
        if (self.current_image_path and 
            os.path.basename(self.current_image_path) == filename):
            self.image_display.config(image='')
            self.clear_analysis_results()
            self.current_image_path = None

    def on_row_select(self, event):
        selected = self.data_table.focus()
        if not selected:
            return
            
        filename = self.data_table.item(selected, "values")[0]
        filepath = os.path.join(IMAGE_FOLDER, filename)
        
        if os.path.exists(filepath):
            self.current_image_path = filepath
            self.display_selected_image(filepath)
            self.clear_analysis_results()

    def clear_analysis_results(self):
        self.gender_result.set("Gender: -")
        self.age_result.set("Age Range: -")
        self.emotion_result.set("Emotion: -")
        self.race_result.set("Race: -")

    def convert_frame_for_display(self, frame):
        """Convert OpenCV frame for Tkinter display"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame).resize((640, 480))
        return ImageTk.PhotoImage(pil_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalysisApp(root)
    root.mainloop()