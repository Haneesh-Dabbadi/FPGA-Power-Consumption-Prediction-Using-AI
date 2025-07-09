# 🔌 FPGA Power Consumption Prediction Using AI

### 👨‍💻 Developed by:
- D. Haneesh 
- D. Boney Rahul
- M. Obaiah

---

## 📘 Project Overview

This project aims to **predict the power consumption of FPGA-based designs using Machine Learning algorithms**. FPGAs are widely used for AI, edge computing, automotive, and real-time embedded systems — but their **power optimization is a critical challenge**. Traditional simulation tools like Xilinx Vivado or Intel Quartus are time-consuming and computationally expensive.

💡 **Our solution?**  
Use **AI-based regression models** to estimate power consumption **early in the design cycle**, thereby helping engineers make **informed, power-aware design decisions** without full hardware synthesis.

---

## 🎯 Objectives

- Predict FPGA power usage **before full synthesis**
- Reduce design time and simulation overhead
- Assist in **real-time power optimization**
- Compare performance of multiple ML models to find the most accurate predictor

---

## 📊 Methodology

### 🔧 Input Features:
- Logic Utilization
- Bandwidth
- I/O Pin Usage
- Configurable Logic Blocks (CLBs)
- Clock Frequency
- Resource Allocation

### 🤖 Algorithms Used:
- Linear Regression
- K-Nearest Neighbors (KNN) Regressor
- **Decision Tree Regressor (Best Performing)**

### 🛠 Tools & Technologies:
- Python (for ML model training)
- Scikit-learn (ML models)
- Tkinter (GUI for prediction system)
- Jupyter Notebook
- Pandas, NumPy, Matplotlib
- Open-source FPGA design datasets

---

## 🧪 Results

| Algorithm              | MAE     | MSE    | RMSE    | Accuracy ↑ |
|------------------------|-----------|-----------|-----------|------------|
| Linear Regressor       | High  | High  | High  | ~4%       |
| KNN Regressor          | Moderate      | High       | High       | ~42%       |
| **Decision Tree (Best)** | **Very Low** | **Very Low** | **Very Low** | **98%+**   |

📈 The **Decision Tree Regressor** outperformed others in terms of **accuracy, simplicity, and computational speed**.

---

## 🖥️ System Interface

- 📥 Admin and User modes
- 📂 Upload dataset CSV file
- 🔄 View dataset analytics
- 🧠 Predict power from design specs
- 📊 Visual comparison of models

> GUI developed using **Tkinter** for an intuitive user experience.

---

## ✅ Key Benefits

- ⚡ **Energy-efficient FPGA designs**
- 🧠 **AI-driven early prediction**
- 🔁 **Reduces dependence on simulation tools**
- 📉 **Lowers cost and power overhead**
- 📦 **Scalable for future FPGA technologies**

---

## 🌐 Applications

- AI Accelerators  
- Edge Computing & IoT  
- Aerospace & Defense  
- Medical Imaging  
- Automotive Systems  
- Cloud Infrastructure  
- Wearable Devices  

---

## 🔮 Future Scope

- Deploy prediction system on **cloud-based FPGA tools**
- Integrate **Reinforcement Learning** for dynamic power control
- Extend to **real-time power monitoring systems**
- Explore **federated ML models** for distributed FPGA systems
