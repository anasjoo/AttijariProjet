# Pipeline CRISP-DM - Détection d'Anomalies Bancaires
# Script PowerShell pour exécuter le pipeline complet

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PIPELINE CRISP-DM - DETECTION D'ANOMALIES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Yellow
& "..\attijarivenv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Test du pipeline..." -ForegroundColor Yellow
python test_pipeline.py

Write-Host ""
Write-Host "Execution du pipeline complet..." -ForegroundColor Yellow
python crisp_dm_pipeline.py

Write-Host ""
Write-Host "Pipeline terminé!" -ForegroundColor Green
Read-Host "Appuyez sur Entrée pour continuer"
