@echo off
echo Cleaning up .pyc files and __pycache__ directories...

:: Remove .pyc files in the repository and subdirectories
for /R %%i in (*.pyc) do (
    echo Deleting %%i
    del /Q "%%i"
)

:: Remove __pycache__ directories in the repository and subdirectories
for /d /r . %%d in (__pycache__) do (
    echo Removing directory %%d
    rmdir /s /q "%%d"
)

echo Cleanup completed.
pause
