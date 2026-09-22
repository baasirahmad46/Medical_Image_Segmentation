@echo off
call C:\ProgramData\anaconda3\Scripts\activate.bat
call conda activate tf210
F:
cd F:\PROJECT\college_Project\Medical_Image_Segmentation
python -m streamlit run app.py
pause