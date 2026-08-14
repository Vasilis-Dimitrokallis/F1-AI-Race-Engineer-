# F1 AI Race Engineer Agent 🏎️

An AI-powered race strategy system for Formula 1. It combines machine learning models trained on real telemetry data with a conversational agent that can reason about strategy, tire wear, and FIA regulations in natural language.

Built as a group project applying Machine Learning and AI techniques to a real-world, high-stakes decision problem: helping an F1 team optimize pit-stop timing and resource management to minimize cost and time lost outside of racing.

## Demo

📹 A demo video walking through the system is available [here](https://youtu.be/BQNZ9egJEEU) *(English subbed)*.
📊 The project presentation is included in this repo: [`presentation.pptx`](./presentation.pptx) *(in Greek)*.

## Overview

The project is structured in four main parts:

1. **Pit-Stop Classifier** — predicts whether a car should pit on a given lap
2. **Tire Wear / Remaining Life Model** — predicts how many laps of life remain in a tire set
3. **Circuit Risk Dashboards** — visualizes high-risk zones on a track based on marshal incident data
4. **Conversational AI Agent** — a natural-language interface that ties all of the above together, backed by a RAG pipeline over the official FIA regulations

## Data

All data is sourced from the [FastF1](https://docs.fastf1.dev/) Python library, which provides official F1 timing, telemetry, and session data. For each session we use:

- **Lap data** — compound, tire life, lap time, track status, driver
- **Telemetry** — throttle, brake, speed, gear, RPM (sampled dozens of times per lap)
- **Weather data** — track temperature, air temperature, rainfall
- **Race control messages** — safety cars, VSC, red/yellow flags, track limit violations, incidents

## Methodology

### Feature Engineering
Telemetry is sampled far more frequently than lap time is recorded, so raw signals cannot be used directly. Features are engineered through **data binning** (e.g. percentage of time throttle is above 90%) to summarize telemetry behavior at the lap level and correlate it with tire wear and performance.

### Pit-Stop Classifier
Since no strong linear relationships were found between features (checked via correlation heatmaps across circuits, drivers, and laps), a **Random Forest classifier** was chosen for its interpretability and robustness to non-linear relationships. Pit stops are naturally rare events, so class imbalance was addressed with **undersampling** and **decision threshold tuning**, evaluated primarily on **recall** — prioritizing catching real pit-stop situations over avoiding false alarms. The model reaches ~97% accuracy and ~90% recall on the pit-stop class.

An **LSTM** was also tested as a benchmark on sequential lap windows.

### Tire Wear Prediction
An **XGBoost regression** model estimates the remaining useful life (RUL) of a tire set based on accumulated wear indicators, evaluated with R² and adjusted R².

### Circuit Risk Dashboards
Marshal race control messages are parsed and classified (safety cars, VSC, track limit violations, driver errors) and located along the circuit using turn/sector geometry extracted from a clean reference lap. Incidents are aggregated into a smoothed risk profile and rendered as a 6-panel dashboard per circuit, including a ranking of drivers most frequently involved in accidents or track-limit violations.

### Conversational Agent (RAG)
The FIA regulations PDF is cleaned, chunked, embedded, and stored in a local vector database. An agent built on the **Google Gemini API** is instructed to act as a Chief Race Engineer, with access to three tools:
- Query the telemetry/ML models for a given driver, lap, and circuit
- Search the FIA regulations via RAG
- Generate and summarize a circuit's risk dashboard

The agent decides which tool(s) to call based on the user's natural-language question and responds accordingly.

## Tech Stack

- **Data & ML:** Python, FastF1, pandas, NumPy, scikit-learn (Random Forest, undersampling, threshold tuning), XGBoost, TensorFlow/Keras (LSTM)
- **Visualization:** Matplotlib, Seaborn
- **NLP / Agent:** Google Gemini API, RAG (text chunking, embeddings, vector database)
- **Notebook:** Jupyter

## Project Structure

```
├── Project.ipynb        # Full notebook: EDA, modeling, dashboards, agent
├── presentation.pdf      # Project presentation (in Greek)
├── README.md
```

## Getting Started

1. Clone the repo and install dependencies:
   ```bash
   pip install fastf1 pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn google-generativeai
   ```
2. Open `Project.ipynb` in Jupyter.
3. Set your own Google Gemini API key as an environment variable before running the agent cells:
   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```
4. Run the notebook top to bottom. Note: fetching multi-season FastF1 data can take a while on first run (cached afterward).

## Notes

This was developed as an academic group project to explore how ML and generative AI can support real-time decision-making in a technical, data-rich domain like Formula 1. It is not affiliated with the FIA or any F1 team.
