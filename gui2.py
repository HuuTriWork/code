import tkinter as tk
from tkinter import ttk
import time

def create_update_notification():
    # Tạo cửa sổ chính
    root = tk.Tk()
    root.title("Thông báo cập nhật")
    root.geometry("600x400")
    root.configure(bg="#f0f0f0")
    
    # Đặt cửa sổ luôn ở trên cùng
    root.attributes('-topmost', True)
    
    # Tạo style cho giao diện
    style = ttk.Style()
    style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
    style.configure("Title.TLabel", background="#f0f0f0", font=("Arial", 16, "bold"))
    style.configure("TFrame", background="#f0f0f0")
    
    # Tạo main frame
    main_frame = ttk.Frame(root, padding="20", style="TFrame")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Tiêu đề
    title_label = ttk.Label(main_frame, 
                           text="🔔 THÔNG BÁO CẬP NHẬT HỆ THỐNG", 
                           style="Title.TLabel")
    title_label.pack(pady=(0, 20))
    
    # Nội dung thông báo
    message_text = "Hệ thống đang được cập nhật và nâng cấp để phục vụ bạn tốt hơn."
    message_label = ttk.Label(main_frame, 
                             text=message_text,
                             wraplength=400,
                             justify=tk.CENTER)
    message_label.pack(pady=(0, 10))
    
    # Thời gian dự kiến
    time_text = "⏰ Thời gian dự kiến hoàn thành:\n20h00 ngày 23/8/2025"
    time_label = ttk.Label(main_frame, 
                          text=time_text,
                          font=("Arial", 14, "bold"),
                          foreground="#d35400",
                          justify=tk.CENTER)
    time_label.pack(pady=(10, 20))
    
    # Thông báo cảm ơn
    thank_you_text = "Cảm ơn sự kiên nhẫn và ủng hộ của quý khách! 💖"
    thank_you_label = ttk.Label(main_frame, 
                               text=thank_you_text,
                               font=("Arial", 11, "italic"),
                               foreground="#27ae60",
                               justify=tk.CENTER)
    thank_you_label.pack(pady=(10, 0))
    
    # Hiển thị thời gian thực
    def update_current_time():
        current_time = time.strftime("⏳ Thời gian hiện tại: %H:%M:%S - %d/%m/%Y")
        current_time_label.config(text=current_time)
        root.after(1000, update_current_time)  # Cập nhật mỗi giây
    
    current_time_label = ttk.Label(main_frame, 
                                  text="",
                                  font=("Arial", 10),
                                  foreground="#7f8c8d")
    current_time_label.pack(pady=(20, 0))
    
    # Bắt đầu cập nhật thời gian
    update_current_time()
    
    # Chạy ứng dụng
    root.mainloop()

if __name__ == "__main__":
    create_update_notification()


