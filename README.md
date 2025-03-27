# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
Topic: AI-Driven Hyper-Personalization & Recommendations

User Dashboard name : SynaptiQ Financial Assistant 

SynaptiQ is a revolutionary AI-powered financial assistant that brings hyper-personalized financial insights to individuals and businesses!

SynaptiQ is a **next-generation AI-powered financial intelligence platform**, designed to provide **real-time, AI-driven financial decision-making support**. With **advanced fraud prevention, consent-based AI insights, and a chatbot supporting voice and image inputs**, SynaptiQ sets a **new benchmark for AI-driven financial security and investment management**. With 88% test coverage**, ML-powered risk assessments, and AI-driven **customer retention strategies


## 🎥 Demo - If you tube link not working, try Google drive link

(Consider this as final link unless any issues)Demo link - https://youtu.be/VaeMpktWMws


Backup demo link - https://youtu.be/ZfeX1CWt7l0

Gdrive demo link - https://drive.google.com/file/d/1LMs2giKhYGR9LNHNufkEvCyzFx-y730L/view?usp=drive_link

Gdrive backup demo link - https://drive.google.com/file/d/16v0Qd787tJpOKdffxeYwxcCPZcjROVO1/view?usp=drive_link

🖼️ Screenshots: An example screen which gives insight how we have trained our models and f1 score can be found.

![Screenshot 2025-03-25 000254](https://github.com/user-attachments/assets/ed2a40ae-4f10-4da7-9691-a084ab9bcc2b)


PPT: https://github.com/ewfx/aidhp-synapti-q/tree/main/artifacts/demo

## 💡 Inspiration
In today's fast-paced financial world, individuals and businesses struggle with complex financial decision-making, fraud prevention, and investment planning. Many people lack access to real-time insights, leading to poor financial choices.

We built SynaptiQ Financial Assistant to empower users with AI-driven financial intelligence. Our goal was to:
✅ Simplify financial decision-making with AI-powered recommendations.
✅ Prevent fraud by detecting suspicious transactions in real-time.
✅ Enhance wealth management by providing personalized investment insights.
✅ Reduce customer churn with AI-based predictive analytics.

By leveraging LLMs, NLP, and financial modeling, SynaptiQ provides a hyper-personalized AI assistant that helps users make smarter financial decisions. 🚀

## ⚙️ What It Does
****1. **Personalized Financial Advice**  

2. **Fraud Detection & Prevention**

3. Loan Approval Predictions**

4. Churn Risk Analysis (with Consent-Based AI Decisioning)**

5. Subscription Management Insights**

6.  **Wealth Management Recommendations**

7.  **Business & Market Insights**

8.  AI-Powered Chatbot (with Voice & Image Processing)**

9.  **Social Media-Based Financial Sentiment Analysis****

10.  **It analyzes fianncial imgaes, such bank statements, slary slips or bills and provide insight based out of it******

**Note - ChatBot special features-> here are few examples**

   Suppose user asked Financial advise for a customer with id, CUST-0001 then fiancial insight api returns a hyper personalized response
   On top of this if user wants to know more example or insights for CUST-0001 then user can start discussion with Bot,
      example - > can yo give few example or create tabular example and explain invstment risk for CUST-0001
                  Bot responses and this discussion can go on.

                  Similarly for fraud detection and etc.
   
   This chatbot is super dynamic, it will answer real time financial based questions, and not just answer an user can interact and continue the discussion
      example - User-> tell me how to calculate compund interest
               Bot -> It provides a response with all calculation
               User-> canyou explain with an example
               Bot-> It provides response with example ......and this discussion can go on
   

## 🛠️ How We Built It

Backend Technologies:

FastAPI!

XGBoost

FAISS

Google Gemini API

Torchvision & OpenAI CLIP

Pandas & NumPy

Frontend Technologies:

ReactJS

Redux

Material-UI

Axios

Chart.js

Speech Recognition API

Service Workers

React Router


## 🚧 Challenges We Faced
During the development of SynaptiQ Financial Assistant, we encountered several technical and non-technical challenges that required innovative solutions.

🔹 1. Data Privacy & Security
Ensuring that user financial data remains secure and compliant with industry standards.

Implementing consent-based AI decisioning while maintaining transparency.

🔹 2. Real-Time Fraud Detection & Risk Analysis
Training AI models to accurately predict fraud risk based on financial transactions.

Balancing false positives and false negatives in fraud detection models.

🔹 3. AI-Powered Personalization
Developing a hyper-personalized recommendation engine that adapts to different user profiles.

Fine-tuning LLMs and NLP models to provide context-aware financial advice.

🔹 4. Technical Integrations & Scalability
Integrating multiple AI models (LLMs, XGBoost, FAISS) into a single FastAPI backend.

Ensuring the system remains scalable to handle large amounts of financial data.

🔹 5. User Experience & Adoption
Designing an intuitive UI that makes complex financial insights easy to understand.

Implementing voice & image-based chatbot interactions for seamless user engagement.

Despite these challenges, our team successfully built SynaptiQ Financial Assistant, delivering AI-driven financial intelligence that enhances decision-making and security. 🚀

## 🏃 How to Run
**Frontend Setup (React)**
1. Clone the repository  
   ```sh
   git clone https://github.com/your-repo.git](https://github.com/ewfx/aidhp-synapti-q.git
   ```
2. Install dependencies  
   ```sh
   npm install  # or pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
   npm start  # or python app.py
   ```
   The frontend app will be available at http://localhost:3000
   
**Backend Setup (FastAPI)**

**Note** - Create .env file unser src folder and add below lines
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

API_KEY = "enter your key here"

1. cd aidhp-synapti-q\code
2. Create & Activate a Virtual Environment. Make sure Python is installed( we have build this on python 3.12.0)
   Windows (PowerShell) : run below command
   
   python -m venv venv
   
   Set-ExecutionPolicy Unrestricted -Scope Process
   
   .\venv\Scripts\activate
   
3. Install Dependencies
   
   pip install -r requirements.txt
   
4. Run the Backend Server
   
   uvicorn banking_api:app --host 0.0.0.0 --port 8000 --reload

   The backend API will now be running at http://127.0.0.1:8000.

 Note - Testcases are placed under test folder. We have unit tests for api end points.To run tests case navigate to tests folder after creating virtual environment then **python test_banking_api.py**  
   

## 🏗️ Tech Stack
Backend Technologies:

FastAPI!

XGBoost

FAISS

Google Gemini API

Torchvision & OpenAI CLIP

Pandas & NumPy

Frontend Technologies:

ReactJS

Redux

Material-UI

Axios

Chart.js

Speech Recognition API

Service Workers

React Router

## 👥 Team : SynaptiQ
- **Team** - Vivek Mani,Sonali Dwivedi, Sumabindu, Kusuma

