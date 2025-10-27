@echo off
echo ========================================
echo PIPELINE CRISP-DM - DETECTION D'ANOMALIES
echo ========================================
echo.

echo Activation de l'environnement virtuel...
call ..\attijarivenv\Scripts\activate.bat

echo.
echo Test du pipeline...
python test_pipeline.py

echo.
echo Execution du pipeline complet...
python crisp_dm_pipeline.py

echo.
echo Pipeline termine!
pause
