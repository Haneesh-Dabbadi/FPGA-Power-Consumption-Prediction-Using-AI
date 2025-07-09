from tkinter import *
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global Variables
data = None
x_train, x_test, y_train, y_test = None, None, None, None
model = None

# Professional Color Palette
COLORS = {
    'background': '#F5F5F5',     # Light gray background
    'primary': '#2C3E50',        # Deep navy blue for headers
    'secondary': '#3498DB',      # Bright blue for accents
    'accent_1': '#E74C3C',       # Vibrant red
    'accent_2': '#2ECC71',       # Fresh green
    'text_dark': '#2C3E50',      # Dark text color
    'text_light': '#FFFFFF',     # Light text color
    'button_hover': '#34495E',   # Darker blue-gray for hover
}

# Function to Upload Dataset
def uploadDataset():
    global data
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    
    if not filename:
        messagebox.showwarning("Warning", "No file selected!")
        return
    
    data = pd.read_csv(filename)
    text.insert(END, "Dataset Loaded Successfully!\n")
    text.insert(END, str(data.head()) + "\n")

# Function to Preprocess Dataset
def preprocessDataset():
    global data, x_train, x_test, y_train, y_test

    if data is None:
        messagebox.showwarning("Warning", "Please upload dataset first!")
        return

    text.delete('1.0', END)
    data.dropna(inplace=True)

    # Encoding categorical columns
    le = LabelEncoder()

    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])

    # Splitting dataset
    X = data.drop(['PowerConsumption'],axis=1)
    y = data['PowerConsumption']
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=77)

    text.insert(END, "Preprocessing Completed!\n")
    text.insert(END, f"Training Data: {x_train.shape}\n")
    text.insert(END, f"Testing Data: {x_test.shape}\n")


# Function to Calculate Regression Metrics
def calculateRegressionMetrics(algorithm, predict, testY):
    mae = mean_absolute_error(testY, predict)
    mse = mean_squared_error(testY, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predict)
    
    text.insert(END, f"{algorithm}\n")
    text.insert(END, f"Mean Absolute Error (MAE): {mae:.2f}%\n")
    text.insert(END, f"Mean Squared Error (MSE): {mse:.2f}%\n")
    text.insert(END, f"Root Mean Squared Error (RMSE): {rmse:.2f}%\n")
    text.insert(END, f"R2 Score: {r2:.2f}%\n")
    
    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(7, 7))
    plt.scatter(testY, predict, color='blue', alpha=0.6)
    plt.plot([min(testY), max(testY)], [min(testY), max(testY)], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{algorithm} - Predicted vs Actual Values")
    plt.grid(True)
    plt.show()

# Function to Train Linear Regression
def trainLR():
    global model
    text.delete('1.0', END)
    if x_train is None or y_train is None:
        messagebox.showwarning("Warning", "Preprocess dataset first!")
        return

    modelfile = 'model/LR.pkl'
    if os.path.exists(modelfile):
        model = joblib.load(modelfile)
        text.insert(END, "Linear Regression Model Loaded Successfully.\n")
    else:
        model = LinearRegression()
        model.fit(x_train, y_train)
        joblib.dump(model, modelfile)
        text.insert(END, "Linear Regression Model Trained and Saved.\n")

    predict = model.predict(x_test)
    calculateRegressionMetrics("Linear Regression", predict, y_test)

# Function to Train KNN Regressor
def trainKNNRegressor():
    global knn_regressor
    text.delete('1.0', END)
    if x_train is None or y_train is None:
        messagebox.showwarning("Warning", "Preprocess dataset first!")
        return

    modelfile = 'model/knn_regressor.pkl'
    if os.path.exists(modelfile):
        knn_regressor = joblib.load(modelfile)
    else:
        knn_regressor = KNeighborsRegressor()
        knn_regressor.fit(x_train, y_train)
        joblib.dump(knn_regressor, modelfile)

    predict = knn_regressor.predict(x_test)
    calculateRegressionMetrics("KNN Regressor", predict, y_test)

# Function to Train Decision Tree Regressor
def trainDecisionTreeRegressor():
    global dt_regressor
    text.delete('1.0', END)
    if x_train is None or y_train is None:
        messagebox.showwarning("Warning", "Preprocess dataset first!")
        return

    modelfile = 'model/DTR.pkl'
    if os.path.exists(modelfile):
        dt_regressor = joblib.load(modelfile)
        text.insert(END, "Model loaded successfully.")
    else:
        dt_regressor = DecisionTreeRegressor(random_state=42)
        dt_regressor.fit(x_train, y_train)
        joblib.dump(dt_regressor, modelfile) 
        text.insert(END, "Model saved successfully.")
    predict = dt_regressor.predict(x_test)
    calculateRegressionMetrics("Decision Tree Regressor", predict, y_test)

# Function to Predict from Test Data
def predictFromTestData():
    text.delete('1.0', END)
    if 'dt_regressor' not in globals():
        messagebox.showwarning("Warning", "Train the model first!")
        return
    
    filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    
    if not filename:
        messagebox.showwarning("Warning", "No file selected!")
        return
    
    test_data = pd.read_csv(filename)
    test_data.dropna(inplace=True)
    
    # Display expected features
    text.insert(END, "Expected Input Features:\n")
    text.insert(END, ", ".join(test_data.columns) + "\n\n")
    
    pred = dt_regressor.predict(test_data)
    for i, p in enumerate(pred):
        text.insert(END, f"Input Features: {test_data.iloc[i].to_dict()}\n")
        text.insert(END, f"Row {i}: Predicted Power Consumption: {p:.2f}\n\n")

# Function to Create Manual Prediction Form
def createManualPredictionForm():
    # Check if model is trained
    if 'dt_regressor' not in globals():
        messagebox.showwarning("Warning", "Train the model first!")
        return

    # Create a new top-level window for manual input
    manual_input_window = tk.Toplevel(main)
    manual_input_window.title("Manual Power Consumption Prediction")
    manual_input_window.geometry("600x700")
    manual_input_window.configure(bg=COLORS['background'])

    # Title for the manual input form
    title_label = tk.Label(
        manual_input_window, 
        text="Enter Feature Values for Prediction", 
        font=('Segoe UI', 16, 'bold'), 
        bg=COLORS['background'], 
        fg=COLORS['primary']
    )
    title_label.pack(pady=20)

    # Create input fields dynamically based on training data columns
    input_entries = {}
    
    # Remove 'PowerConsumption' column if present
    feature_columns = [col for col in x_train.columns]
    
    for column in feature_columns:
        frame = tk.Frame(manual_input_window, bg=COLORS['background'])
        frame.pack(fill=X, padx=50, pady=5)
        
        label = tk.Label(
            frame, 
            text=column, 
            font=('Segoe UI', 12), 
            bg=COLORS['background'], 
            fg=COLORS['text_dark']
        )
        label.pack(side=LEFT, padx=10)
        
        entry = tk.Entry(
            frame, 
            font=('Segoe UI', 12), 
            width=30
        )
        entry.pack(side=RIGHT, padx=10)
        input_entries[column] = entry

    def predict_manual_input():
        # Collect input values
        input_data = {}
        for column, entry in input_entries.items():
            try:
                input_data[column] = float(entry.get())
            except ValueError:
                messagebox.showerror("Error", f"Invalid input for {column}. Please enter a numeric value.")
                return
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict using the trained Decision Tree Regressor
        prediction = dt_regressor.predict(input_df)
        
        # Clear previous text and show prediction
        text.delete('1.0', END)
        text.insert(END, "Manual Input Prediction:\n")
        text.insert(END, "Input Features:\n")
        for column, value in input_data.items():
            text.insert(END, f"{column}: {value}\n")
        text.insert(END, f"\nPredicted Power Consumption: {prediction[0]:.2f}\n")
        
        # Close the manual input window
        manual_input_window.destroy()

    # Prediction Button
    predict_button = tk.Button(
        manual_input_window, 
        text="Predict", 
        command=predict_manual_input,
        font=('Segoe UI', 12, 'bold'),
        bg=COLORS['secondary'],
        fg=COLORS['text_light'],
        padx=20,
        pady=10
    )
    predict_button.pack(pady=20)

# Function to Exit Application
def exitApp():
    main.destroy()

# Customized Gradient Background Function
def create_gradient_background(canvas, width, height):
    # Create a gradient from light blue to white
    for i in range(height):
        # Interpolate color from top to bottom
        r = int(135 + (245 - 135) * (i / height))
        g = int(206 + (245 - 206) * (i / height))
        b = int(235 + (245 - 235) * (i / height))
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i, width, i, fill=color)

# Customized Button Creation
def create_stylish_button(parent, text, command, bg_color, hover_color):
    # Create a frame for 3D effect
    btn_frame = Frame(parent, bg=COLORS['background'])
    
    # Button with gradient and shadow effect
    btn = tk.Button(
        btn_frame, 
        text=text, 
        command=command,
        font=('Segoe UI', 12, 'bold'),
        bg=bg_color,
        fg=COLORS['text_light'],
        activebackground=hover_color,
        activeforeground=COLORS['text_light'],
        relief=RAISED,
        borderwidth=0,
        padx=15,
        pady=8
    )
    
    # Hover effects
    def on_enter(e):
        btn.config(bg=hover_color)
        btn_frame.config(bg=COLORS['secondary'])
    
    def on_leave(e):
        btn.config(bg=bg_color)
        btn_frame.config(bg=COLORS['background'])
    
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    btn.pack(expand=True, fill=BOTH, padx=2, pady=2)
    
    return btn_frame

# Initialize Tkinter Window
main = Tk()
main.title("FPGA Power Consumption Prediction")
main.state('zoomed')  # Maximize window

# Create canvas for gradient background
canvas = Canvas(main, highlightthickness=0)
canvas.pack(fill=BOTH, expand=True)

# Create gradient background
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
create_gradient_background(canvas, screen_width, screen_height)

# Main Frame
main_frame = Frame(canvas, bg='white')
canvas.create_window((0, 0), window=main_frame, anchor='nw', width=screen_width, height=screen_height)

# Title Frame with Gradient
title_frame = Frame(main_frame, bg=COLORS['primary'])
title_frame.pack(fill=X, pady=(0,20))

title = Label(
    title_frame, 
    text='FPGA Power Consumption Prediction using AI', 
    font=('Segoe UI', 22, 'bold'), 
    bg=COLORS['primary'], 
    fg='white',
    padx=20,
    pady=15
)
title.pack()

# Mode Selection Frame
mode_frame = Frame(main_frame, bg='white')
mode_frame.pack(pady=20)

# Button Container Frame
button_container = Frame(main_frame, bg='white')
button_container.pack(expand=True, fill=X, pady=20)

# Create a frame to perfectly center and align buttons
button_frame = Frame(button_container, bg='white')
button_frame.pack(expand=True)

# Mode Selection Buttons
def showAdminButtons():
    # Clear previous grid
    for widget in button_frame.winfo_children():
        widget.grid_forget()
    
    # Place buttons in a single row
    btn_upload.grid(row=0, column=0, padx=10, pady=5)
    btn_preprocess.grid(row=0, column=1, padx=10, pady=5)
    btn_LR.grid(row=0, column=2, padx=10, pady=5)
    btn_KNN.grid(row=0, column=3, padx=10, pady=5)
    btn_DTR.grid(row=0, column=4, padx=10, pady=5)

def showUserButtons():
    # Clear previous grid
    for widget in button_frame.winfo_children():
        widget.grid_forget()
    
    # Center user buttons
    btn_predict.grid(row=0, column=0, padx=10, pady=5)
    btn_manual_predict.grid(row=0, column=1, padx=10, pady=5)
    btn_exit.grid(row=0, column=2, padx=10, pady=5)

# Create stylish mode selection buttons
admin_button = create_stylish_button(
    mode_frame, 
    "ADMIN MODE", 
    showAdminButtons, 
    COLORS['secondary'], 
    COLORS['button_hover']
)
admin_button.pack(side=LEFT, padx=20)

user_button = create_stylish_button(
    mode_frame, 
    "USER MODE", 
    showUserButtons, 
    COLORS['accent_2'], 
    COLORS['accent_1']
)
user_button.pack(side=LEFT, padx=20)

# Admin Buttons with stylish design
btn_upload = create_stylish_button(
    button_frame, 
    "Upload Dataset", 
    uploadDataset, 
    COLORS['secondary'], 
    COLORS['button_hover']
)
btn_preprocess = create_stylish_button(
    button_frame, 
    "Preprocessing", 
    preprocessDataset, 
    COLORS['accent_2'], 
    COLORS['accent_1']
)

btn_LR = create_stylish_button(
    button_frame, 
    "Linear Regression", 
    trainLR, 
    COLORS['secondary'], 
    COLORS['button_hover']
)
btn_KNN = create_stylish_button(
    button_frame, 
    "KNN Regressor", 
    trainKNNRegressor, 
    COLORS['accent_2'], 
    COLORS['accent_1']
)
btn_DTR = create_stylish_button(
    button_frame, 
    "Decision Tree", 
    trainDecisionTreeRegressor, 
    COLORS['secondary'], 
    COLORS['button_hover']
)

# User Buttons
btn_predict = create_stylish_button(
    button_frame, 
    "Predict from Test Data", 
    predictFromTestData, 
    COLORS['accent_2'], 
    COLORS['accent_1']
)
btn_manual_predict = create_stylish_button(
    button_frame, 
    "Manual Prediction", 
    createManualPredictionForm, 
    COLORS['secondary'], 
    COLORS['button_hover']
)
btn_exit = create_stylish_button(
    button_frame, 
    "EXIT", 
    exitApp, 
    COLORS['accent_1'], 
    COLORS['primary']
)

# Output Textbox with modern styling
text = Text(
    main_frame, 
    height=20, 
    width=150, 
    font=('Consolas', 12),  
    bg='#2C3E50',  # Dark background
    fg='#2ECC71',  # Bright green text
    wrap=WORD,
    padx=15,
    pady=10,
    insertbackground='white'  # Cursor color
)
text.pack(pady=20)

# Run Tkinter Main Loop
main.mainloop()