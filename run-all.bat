@echo off
setlocal enabledelayedexpansion

set "MASK=.\random_tilt\compound_rotation*.png"
set "RESULT_DIR=.\random_tilt_result"

echo Starting batch processing for mask: %MASK%
echo ----------------------------------------------------

if not exist "%RESULT_DIR%" (
    mkdir "%RESULT_DIR%"
)

for %%F in (%MASK%) do (
    set "FILENAME=%%~nF"
    echo !FILENAME!| findstr /R /C:"-debug$" /C:"-stat$"   
    if !errorlevel! neq 0 (
        set "LOGFILE=!RESULT_DIR!\!FILENAME!.txt"    
        echo Processing: %%F -^> !LOGFILE!    
        python detector.py -V -C --input "%%F" --output "!RESULT_DIR!" > "!LOGFILE!"
    )
)

echo ----------------------------------------------------
echo Done! All logs collected.
pause
endlocal
