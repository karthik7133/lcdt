import customtkinter as ctk

def dismiss_alert():
    """Closes the UI and terminates the script."""
    app.destroy()

# Set modern dark mode theme
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("Digital Twin Alert")

# 1. Strip the window borders and title bar
app.overrideredirect(True)

# 2. Force the window to float above all other apps
app.attributes("-topmost", True)

# 3. Apply the glassmorphism effect (85% opacity)
app.attributes("-alpha", 0.90)

# Make the outer container completely transparent so we only see the rounded frame
app.configure(fg_color="#000000") 

# 4. Calculate coordinates to perfectly center the popup
window_width = 500
window_height = 250
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
app.geometry(f"{window_width}x{window_height}+{x}+{y}")

# --- NEON GLOSSY FRAME ---
# Deep sleek purple/blue background with neon cyan border
main_frame = ctk.CTkFrame(
    app, 
    width=window_width, 
    height=window_height, 
    corner_radius=25,       # Beautiful rounded modern edges
    fg_color="#0A0B10",     # Deep obsidian background
    border_width=2,
    border_color="#00FFFF"  # Neon Cyan glowing border
)
main_frame.pack(fill="both", expand=True, padx=2, pady=2)
main_frame.pack_propagate(False)

# --- UI ELEMENTS ---
# Warning Title with Neon Red/Pink
title_label = ctk.CTkLabel(
    main_frame, 
    text="⚠️ CYBER RISK CRITICAL", 
    font=("Segoe UI", 24, "bold"), 
    text_color="#FF0055"  # Neon pink/red 
)
title_label.pack(pady=(35, 15))

# Informational Message
msg_label = ctk.CTkLabel(
    main_frame, 
    text="You seem fatigued. Your cyber risk profile is vulnerable.\nInitiate a mandatory 10-minute system rest.", 
    font=("Segoe UI", 15), 
    text_color="#D1D4FF", # Soft ice blue text for readability against the dark
    justify="center"
)
msg_label.pack(pady=(0, 35))

# Neon Green Acknowledge Button
dismiss_btn = ctk.CTkButton(
    main_frame, 
    text="ACKNOWLEDGE & SECURE SYSTEM", 
    font=("Segoe UI", 13, "bold"), 
    fg_color="transparent",       # Hollow "Ghost" button style
    text_color="#39FF14",         # Neon Green text
    border_color="#39FF14",       # Neon Green border
    hover_color="#103A0A",        # Dark green background on hover
    border_width=2,
    corner_radius=12,             # Rounded pill button
    width=260,
    height=45,
    cursor="hand2",
    command=dismiss_alert
)
dismiss_btn.pack()

# Start the UI loop
app.mainloop()
