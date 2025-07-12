# AI-Document-Reader

1. To install Dependencies:
    - Run Following Commands
    -  python -m venv venv : To create Python virtual env
    - .\venv\Scripts\activate : To Activate Venv
    - pip install -r requirements.txt : To install Dependency package

2. To Start the project:
    - Run Command 
    - uvicorn app.main:app --reload
    - Copy the localhost link and paste it to browser
    - http://127.0.0.1:8000/docs : link to paste

3. Till now only PDF file is acceptable

4. Use /upload route to upload the PDF Document