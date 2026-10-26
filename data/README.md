# Dataset Documentation: Mall Customer Segmentation Data

## 1. Primary Dataset: Mall_Customers.csv
The primary dataset utilized in this project is the **Mall Customer Segmentation Data**, widely available on Kaggle and public machine learning repositories.

- **Target Path:** `data/Mall_Customers.csv`
- **Total Records:** 200 customers
- **Attributes:** 
  1. `CustomerID`: Unique numerical ID assigned to each customer (excluded from clustering as it carries no behavioral signal).
  2. `Gender` (or `Genre` in some versions): Categorical gender (Male / Female).
  3. `Age`: Customer age in years (18–70).
  4. `Annual Income (k$)`: Self-reported annual income in thousands of dollars (15–137 k$).
  5. `Spending Score (1-100)`: Behavioral score assigned by the mall based on purchase frequency and transaction totals (1–99).

### Manual Download Steps (if downloading from Kaggle):
1. Visit the Kaggle dataset page: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
2. Click **Download (1.56 kB)**.
3. Extract `Mall_Customers.csv` from the downloaded archive.
4. Place the file inside the project directory at `data/Mall_Customers.csv`.

---

## 2. Fallback Synthetic Sample: sample_synthetic.csv
To guarantee that the application and test suites run completely offline even if `Mall_Customers.csv` is missing:
- A synthetic sample of 200 simulated customers is generated at `data/sample_synthetic.csv`.
- This file is clearly labeled as synthetic demo data and is never presented as authentic Kaggle results.
