# Insurance Charges Prediction with Fast API 

This project is a RESTful API built with FastAPI that predicts medical insurance charges based on personal attributes. It utilizes three pre-trained machine learning models (Random Forest, Ensemble Decision Tree, and XGBoost) to provide predictions.

The application includes robust input validation, custom middleware for performance tracking, and a logging system for request analytics.

## Features

*   **Multi-Model Support:** Choose between Random Forest, Decision Tree, or XGBoost for predictions.
*   **Strict Validation:** Pydantic models ensure data integrity (e.g., age limits, valid BMI ranges, correct One-Hot Encoding sums).
*   **Analytics Logging:** Custom middleware logs detailed request/response metrics (latency, payload size, status codes) to `analytics.log`.
*   **Performance Headers:** Returns X-Process-Time headers to track processing speed.

## Project Structure

**Important:** Your code expects specific relative paths (`../models/`). Ensure your directory looks like this:

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
