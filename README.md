<h1 align="center">
  <br>
  <a href="https://vantage6.ai"><img src="https://github.com/IKNL/guidelines/blob/master/resources/logos/vantage6.png?raw=true" alt="vantage6" width="400"></a>
</h1>

<h3 align=center> A privacy preserving federated learning solution</h3>

--------------------

# v6-stata-fit-stats-py
Computes STRATA-FIT specific statistics on consortium datasets via [vantage6](https://vantage6.ai). To learn more about STRATA-FIT please visit [STRATA-FIT.eu](https://strata-fit.eu/).

## Running the Federated KM Mock‑Client Test

### 1. Install Git and Python  
- **Git**: download and install from https://git‑scm.com/downloads  
- **Python 3.10+**: install from https://python.org/downloads  
- Verify installation by opening a terminal (macOS/Linux) or PowerShell (Windows) and running:  
  ```bash
  python --version  
  ```

### 2. Clone the Repository  
- Navigate to the folder where you want the code  
- Run:
  ```
  git clone https://github.com/mdw-nl/strata-fit-v6-stats-py
  ```
- Change into the project directory:  
  ```bash
  cd strata-fit-v6-stats-py
  ```

### 3. Create & Activate a Virtual Environment  
- **macOS/Linux**:  
  ```bash
  python3 -m venv .venv  
  source .venv/bin/activate
  ```  
- **Windows (PowerShell)**:
  ```powershell  
  python -m venv .venv  
  .venv\Scripts\Activate.ps1
  ```

### 4. Install Dependencies  
- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```
- Install requirements:  
  ```bash
  pip install -r requirements.txt
  ```

### 5. Prepare Your Three Node Datasets  
- Place your STRATA-FIT test CSV file in the tests/ folder (e.g. `test_strata.csv` is provided).
- You can also modify `mock_client.py` to point to different CSVs if needed.

### 6. Execute the Mock Client test
- Run:
  ```bash
  python tests/mock_client.py
  ```

You should see INFO logs about function execution, field validation, and privacy-safe output. The mock client simulates a node environment and runs the full data statistics pipeline

---

### Splitting Your Own Data  
If you have one large CSV, split it into three files (e.g. equal row chunks or by patient ID) named alpha.csv, beta.csv, gamma.csv in tests/data/data_times/. The mock client will treat them as three separate nodes.  

This works on any OS. Simply follow these steps to validate your federated KM implementation end‑to‑end.  
------------------------------------
> [vantage6](https://vantage6.ai)
