# app_gui.py
import customtkinter as ctk
import threading
import queue
from live_engine_final import TrafficAnalyzer # Import our engine

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("S-XG-NID Live Intrusion Detection")
        self.geometry(f"{800}x{500}")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Member Variables ---
        self.analysis_thread = None
        self.ui_queue = queue.Queue()
        self.analyzer = TrafficAnalyzer(self.ui_queue)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="S-XG-NID", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.start_button = ctk.CTkButton(self.sidebar_frame, text="Start Capture", command=self.start_capture_event)
        self.start_button.grid(row=1, column=0, padx=20, pady=10)
        
        self.stop_button = ctk.CTkButton(self.sidebar_frame, text="Stop Capture", state="disabled", command=self.stop_capture_event)
        self.stop_button.grid(row=2, column=0, padx=20, pady=10)
        
        # --- Main Textbox for Alerts ---
        self.textbox = ctk.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.textbox.insert("0.0", "System Initialized. Ready to start analysis.\n\n")
        
        # Start checking the queue for messages from the engine
        self.periodic_queue_check()

    def periodic_queue_check(self):
        """Checks the queue for new messages and updates the textbox."""
        while not self.ui_queue.empty():
            message = self.ui_queue.get_nowait()
            self.textbox.insert("end", f"{message}\n")
            self.textbox.see("end") # Auto-scroll to the bottom
        self.after(200, self.periodic_queue_check) # Check again in 200ms

    def start_capture_event(self):
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.textbox.insert("end", "[INFO] Starting traffic analysis engine...\n")
        
        # Create and start the analysis thread
        self.analysis_thread = threading.Thread(target=self.analyzer.start, daemon=True)
        self.analysis_thread.start()

    def stop_capture_event(self):
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.textbox.insert("end", "[INFO] Stopping traffic analysis engine...\n")
        
        # Signal the analyzer to stop
        self.analyzer.stop()

if __name__ == "__main__":
    app = App()
    app.mainloop()