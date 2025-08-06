# AI in Finance – Smart Portfolio Optimization using Deep Reinforcement Learning

A project that uses Deep Reinforcement Learning (DRL) and financial news sentiment to develop intelligent stock trading agents.

---
## Live Demo

A live version of the project dashboard is available here:
**[Coming Soon !!!](https://your-streamlit-app-link.com)**

---
## About The Project

This project aims to create a smart portfolio management system by training a Deep Reinforcement Learning agent in a simulated stock market environment. The agent's decision-making process is enhanced by incorporating sentiment analysis of financial news, allowing it to learn strategies that adapt to both market price movements and public sentiment.

### Key Features
* Custom stock trading environment built with an OpenAI Gym-style interface.
* DRL agent trained using state-of-the-art libraries.
* Sentiment analysis module using FinBERT to process financial news.
* Interactive Streamlit dashboard for performance visualization and analysis.

---
## Technology Stack

* **Python**: Core programming language
* **PyTorch**: For building neural network models
* **Stable-Baselines3**: For DRL agent implementation
* **Transformers (Hugging Face)**: For using the FinBERT model
* **yfinance**: For fetching historical stock market data
* **Streamlit & Plotly**: For creating the interactive web dashboard
* **Pandas & NumPy**: For data manipulation

---
## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.12.7
* pip package manager

### Installation

1.  Clone the repository:
    ```sh
    git clone [https://github.com/your-username/ai-in-finance-drl.git](https://github.com/your-username/ai-in-finance-drl.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd Custom_DRL_Model
    ```
3.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Generate Agent and Sentiment Data**:
    Before launching the dashboard, you must generate the necessary CSV files. Run the following scripts:
    ```sh
    # (Script to run the agent)
    python run_agent_training.py 
    
    # (Script to run sentiment analysis)
    python generate_sentiment_csvs.py
    ```

2.  **Launch the Dashboard**:
    Once the `dashboard_data` and `sentiment_dashboard_data` folders have been created, you can start the Streamlit app:
    ```sh
    streamlit run app.py
    ```

---
## Project Structure
project-root/
│
├── agents/             # DRL agent code
├── env/                # Custom Gym environment (state, reward, actions)
├── sentiment/          # FinBERT-based sentiment analysis code
├── dashboard/          # Streamlit app code for visualization
├── .gitignore              
├── requirement.txt             
└── README.md          

---
## Team and Contributions

* **Praneet**: DRL agent implementation
* **Chinmay**: Environment design and Segment integration
* **Naman**: Sentiment module using FinBERT
* **Himanshu**: Streamlit dashboard and integration

---
## Future Work

* Integrate real-time price feeds for a live environment.
* Deploy the model for paper trading on a live platform.
* Expand backtesting to include other asset classes like cryptocurrencies or ETFs.
