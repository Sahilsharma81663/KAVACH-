# Kavach

Kavach is an AI-powered online exam monitoring and proctoring platform built with Streamlit, SQLite, OpenCV, browser security scripts, live analytics, and PDF reporting.

## What is included

- Student registration with duplicate checks
- Face registration with Haar cascade face detection
- Student login with password and face verification
- Admin login and platform management
- Exam creation and deletion
- Student dashboard with active exam launch
- Live webcam monitoring with suspicious activity alerts
- Browser security monitoring for fullscreen exit and tab switching
- Suspicion score and ML-assisted risk assessment
- Analytics dashboard with interactive charts
- Session-wise PDF report generation

## Project structure

```text
F:\KAVACH
|-- app.py
|-- README.md
|-- requirements.txt
|-- assets/
|   `-- styles/
|-- data/
|-- kavach/
|   |-- auth.py
|   |-- config.py
|   |-- database.py
|   |-- monitoring.py
|   |-- reporting.py
|   |-- ui.py
|   |-- vision.py
|   `-- components/
|       `-- browser_security/
|-- reports/
`-- tests/
```

## Default admin credentials

- Username: `admin`
- Password: `Admin@123`

## Run locally

1. Install the dependencies from `requirements.txt`.
2. Start the Streamlit app:

```powershell
streamlit run app.py
```

3. Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Notes

- Phone detection is implemented with an optional YOLOv8 path. If `ultralytics` or model weights are unavailable, the rest of the platform still runs and the app surfaces the detector status.
- Optional packages in `.vendor` are only used when they match the running Python interpreter. This prevents ABI mismatches between different Python installs on the same machine.
- The automated tests use isolated temporary storage and do not touch the main app database.
