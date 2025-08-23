import tkinter as tk
from tkinter import ttk
import time
from PIL import Image, ImageTk
import sys

class UpdateNotification:
    def __init__(self, root):
        self.root = root
        self.root.title("Thông báo cập nhật hệ thống")
        self.root.geometry("600x400")
        self.root.configure(bg='#f0f8ff')
        self.root.resizable(False, False)
        
        # Center the window
        self.center_window()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start clock update
        self.update_clock()
    
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="THÔNG BÁO CẬP NHẬT HỆ THỐNG", 
                               font=("Arial", 18, "bold"),
                               foreground="#2c3e50",
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Decorative line
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Info text
        info_text = "Hệ thống đang được cập nhật và nâng cấp để phục vụ bạn tốt hơn."
        info_label = ttk.Label(main_frame, 
                              text=info_text,
                              font=("Arial", 12),
                              wraplength=500,
                              justify=tk.CENTER,
                              style='Info.TLabel')
        info_label.pack(pady=15)
        
        # Expected time frame
        time_frame = ttk.Frame(main_frame, style='Time.TFrame')
        time_frame.pack(pady=20)
        
        time_title = ttk.Label(time_frame, 
                              text="Thời gian dự kiến hoàn thành:",
                              font=("Arial", 14, "bold"),
                              style='TimeTitle.TLabel')
        time_title.pack()
        
        self.time_value = ttk.Label(time_frame, 
                                   text="20h00 ngày 23/8/2025",
                                   font=("Arial", 16, "bold"),
                                   foreground="#e74c3c",
                                   style='TimeValue.TLabel')
        self.time_value.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, 
                                       orient=tk.HORIZONTAL, 
                                       length=400, 
                                       mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start(15)
        
        # Current time
        self.clock_label = ttk.Label(main_frame, 
                                    font=("Arial", 10),
                                    style='Clock.TLabel')
        self.clock_label.pack(pady=10)
        
        # Status message
        self.status_label = ttk.Label(main_frame, 
                                     text="Đang cập nhật...",
                                     font=("Arial", 11),
                                     style='Status.TLabel')
        self.status_label.pack()
        
        # Close button
        close_btn = ttk.Button(main_frame, 
                              text="Đóng", 
                              command=self.root.destroy,
                              style='Close.TButton')
        close_btn.pack(pady=20)
        
        # Configure styles
        self.configure_styles()
    
    def configure_styles(self):
        style = ttk.Style()
        
        # Configure styles
        style.configure('Main.TFrame', background='#f0f8ff')
        style.configure('Title.TLabel', background='#f0f8ff')
        style.configure('Info.TLabel', background='#f0f8ff')
        style.configure('Time.TFrame', background='#e3f2fd', relief='solid')
        style.configure('TimeTitle.TLabel', background='#e3f2fd')
        style.configure('TimeValue.TLabel', background='#e3f2fd')
        style.configure('Clock.TLabel', background='#f0f8ff', foreground='#7f8c8d')
        style.configure('Status.TLabel', background='#f0f8ff', foreground='#27ae60')
        
        # Button style
        style.configure('Close.TButton', 
                       font=('Arial', 10, 'bold'),
                       padding=10,
                       foreground='white',
                       background='#3498db')
        style.map('Close.TButton', 
                 background=[('active', '#2980b9')])
    
    def update_clock(self):
        """Update the current time display"""
        current_time = time.strftime("%H:%M:%S - %d/%m/%Y")
        self.clock_label.config(text=f"Thời gian hiện tại: {current_time}")
        self.root.after(1000, self.update_clock)

def main():
    root = tk.Tk()
    app = UpdateNotification(root)
    root.mainloop()

if __name__ == "__main__":
    main()
