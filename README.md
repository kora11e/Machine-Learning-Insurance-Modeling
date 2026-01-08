# ML2 Final projects 

## Insurance Charges Prediction with Fast API

Authors:
- Karol Rochalski
- Jamal Al-Shaweai

Run via venv for notebook models or dockerfile for API

```python
docker pull karolrochalski/insurance-api:latest
```

### Regression Model

This notebook focuses on building and evaluating regression models to predict target outcomes based on input features using machine learning techniques. The project explores several regression algorithms — including Linear Regression Deep Neural Network with dropout and normalization, Decision Tree Regressor, Random Forest, and XGBoost Regressor — to identify the most effective predictive model.

After extensive feature engineering, hyperparameter optimization, and performance evaluation, the XGBoost Regressor emerged as the best-performing model. It demonstrated strong predictive accuracy and robustness, achieving an R² score of 0.87 and a Mean Absolute Percentage Error (MAPE) of 14% on the test dataset.

These results indicate that the XGBoost model explains approximately 87% of the variance in the target variable and maintains low relative prediction error, showcasing its reliability for real-world forecasting tasks. The notebook includes detailed steps for data preprocessing, model training, performance comparison, and error visualization to ensure transparency and reproducibility.

## Customer Segmentation Prediction for New Markets

This project focuses on building and comparing machine learning models to predict customer segments for a new market. The objective is to help an automobile company classify each potential customer into one of four segments (A, B, C, or D) based on demographic, socioeconomic, and behavioral data. The dataset used for this study includes 8,068 labeled customer records for training and 2,627 unlabeled records for prediction. Key features include variables such as gender, age, marital status, profession, family size, spending score, and education level.

Multiple machine learning algorithms were trained and evaluated to identify the most effective approach for this classification problem. Models tested include Decision Tree, Random Forest, Gradient Boosting, XGBoost, and a Neural Network built with Keras. Among them, the Gradient Boosting Classifier achieved the best validation accuracy of 53.9%, outperforming the other models. A soft-voting ensemble of Random Forest, Gradient Boosting, and XGBoost was also tested but achieved a slightly lower accuracy (52.9%). The neural network model, although competitive, did not surpass the performance of the tree-based methods.

The trained Gradient Boosting model was ultimately selected for final prediction. It was applied to the test dataset to generate predicted customer segments, which were then inverse-transformed into their categorical labels (A, B, C, D). The final output was saved as Predicted_Customer_Segments.csv.

From the analysis, features such as spending score and age emerged as strong predictors of customer segments, while profession and family size contributed nonlinearly. The project demonstrates the application of structured data preprocessing, model comparison, and ensemble learning to a real-world marketing segmentation task.

## API Features

*   Multi-Model Support: Choose between Random Forest, Decision Tree, or XGBoost for predictions.
*   Strict Validation: Pydantic models ensure data integrity (e.g., age limits, valid BMI ranges, correct One-Hot Encoding sums).
*   Analytics Logging: Custom middleware logs detailed request/response metrics (latency, payload size, status codes) to `analytics.log`.
*   Performance Headers: Returns X-Process-Time headers to track processing speed.

## Project Structure

Important: Your code expects specific relative paths (`../models/`). Ensure your directory looks like this:

```text
/project-root
│
├── models/                   # Directory containing pre-trained models
│   ├── rf_regressor.pkl
│   ├── dt_regressor.pkl
│   └── xgb_regressor.pkl
│
├── api/                      # Directory containing your API code
│   └── main.py               # The FastAPI application file
│
├── analytics.log             # Generated automatically at runtime
└── requirements.txt          # Python dependencies
```

## Installation

1.  **Clone the repository** (or set up the folder structure above).
2.  **Install dependencies**. Create a `requirements.txt` with the following libraries (including the specific libraries used to train your pickle models, usually scikit-learn and xgboost):

    ```txt
    fastapi
    uvicorn
    pydantic
    numpy
    scikit-learn
    xgboost
    ```

3.  **Install via pip:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Navigate to the directory containing your `main.py` file and run the Uvicorn server:

```bash
cd src
uvicorn main:app --reload
```

*   The API will be available at: `http://127.0.0.1:8000`
*   Interactive Docs (Swagger UI): `http://127.0.0.1:8000/docs`

## API Endpoints

### 1. Health Check
*   **URL:** `/`
*   **Method:** `GET`
*   **Description:** Returns a welcome message to verify the API is running.

### 2. Predict Charges (Random Forest)
*   **URL:** `/predict_charge/rf`
*   **Method:** `POST`
*   **Description:** Returns prediction using the Random Forest Regressor.

### 3. Predict Charges (Decision Tree)
*   **URL:** `/predict_charge/dt`
*   **Method:** `POST`
*   **Description:** Returns prediction using the Decision Tree Regressor.

### 4. Predict Charges (XGBoost)
*   **URL:** `/predict_charge/xgb`
*   **Method:** `POST`
*   **Description:** Returns prediction using the XGBoost Regressor.

---

## Input Data Schema

All prediction endpoints expect a JSON body with the following structure. The API uses **One-Hot Encoding** for categorical variables.

### JSON Field Details

| Field | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `age` | int | 1 - 120 | Age of the beneficiary. |
| `bmi` | float | 10.0 - 90.0 | Body Mass Index. |
| `children` | int | 0 - 20 | Number of children covered by health insurance. |
| `sex_0`, `sex_1` | float | 0 or 1 | **One-Hot Encoded.** Must sum to exactly 1. |
| `region_0`...`region_3` | float | 0 or 1 | **One-Hot Encoded.** Must sum to exactly 1. |
| `smoker_0`, `smoker_1` | float | 0 or 1 | **One-Hot Encoded.** |

### Example Request Body

```json
{
  "age": 30,
  "bmi": 25.5,
  "children": 2,
  "sex_0": 1.0,
  "sex_1": 0.0,
  "region_0": 0.0,
  "region_1": 1.0,
  "region_2": 0.0,
  "region_3": 0.0,
  "smoker_0": 1.0,
  "smoker_1": 0.0
}
```

### Example Response
```json
{
  "predicted_price": 12345.67
}
```

## Analytics & Logging

The application includes advanced middleware (`analytics_middleware`) that intercepts every request.

1.  **Console/File Logging:**
    Every request is logged to `analytics.log` in JSON format containing:
    *   Timestamp
    *   HTTP Method & Path
    *   Client IP
    *   Processing Time (ms)
    *   Request/Response Size (bytes)

    *Example Log Entry:*
    ```json
    {"timestamp": "2023-10-27T10:00:00.123456", "method": "POST", "path": "/predict_charge/rf", "status_code": 200, "process_time_ms": 45.2, ...}
    ```

2.  **Response Headers:**
    The API appends custom headers to the HTTP response for debugging:
    *   `X-Process-Time`: Time taken to process the request (seconds).
    *   `X-Request-Size`: Size of the incoming body (bytes).
    *   `X-Response-Size`: Size of the outgoing body (bytes).

## Validation Errors

If the input data violates the logic (e.g., `sex_0` and `sex_1` are both 1), the API returns a **422 Unprocessable Entity** error with a descriptive message explaining exactly which validation rule failed.
