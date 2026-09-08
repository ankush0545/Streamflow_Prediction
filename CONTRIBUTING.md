# Contributing to Streamflow Prediction

Thank you for your interest in contributing to the **Streamflow Prediction** project! We welcome contributions ranging from bug fixes, model enhancements, and visualization additions to documentation improvements.

---

## Code of Conduct

Please help us maintain a friendly, welcoming, and collaborative community. Treat all contributors with mutual respect.

---

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ankush0545/Streamflow_Prediction.git
cd Streamflow_Prediction
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Tests
```bash
pytest tests/
# Or using standard python unittest:
python3 -m unittest discover -s tests
```

### 5. Launch the Streamlit App
```bash
streamlit run app/app.py
```

---

## Contribution Workflow

1. **Fork & Branch**: Create a feature branch with a descriptive name:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make Changes**:
   - Adhere to [PEP 8](https://peps.python.org/pep-0008/) style standards.
   - Include docstrings and type annotations for all new classes and functions.
   - Ensure new features include corresponding unit tests in `tests/`.
3. **Run Linting & Tests**:
   - Verify that all unit tests pass before committing.
4. **Submit a Pull Request**:
   - Provide a clear PR title and detailed description of the changes made.
   - Reference any related issues.

---

## Questions or Suggestions?

Feel free to open an issue or reach out via GitHub Discussions. Thank you for contributing!
