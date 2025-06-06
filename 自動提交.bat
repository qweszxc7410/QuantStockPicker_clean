lk@echo off

:: 獲取當前分支名稱
for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set "current_branch=%%i"

:: 確認當前分支是否為master
if /I NOT "%current_branch%"=="master" (
    echo You are not on the master branch. Switching to master branch...
    git checkout -b master
    if errorlevel 1 (
        echo Failed to switch to master branch. Exiting...
        exit /b 1
    )
)

:: 將所有變更的檔案添加到暫存區
echo Adding all changes to staging...
git add -A
if errorlevel 1 (
    echo Failed to add changes to staging. Exiting...
    exit /b 1
)

:: 獲取系統日期和時間
set date_var=%date%
set time_var=%time%

:: 格式化日期和時間為 yyyymmdd_hhmmss
set year=%date_var:~0,4%
set month=%date_var:~5,2%
set day=%date_var:~8,2%
set hour=%time_var:~0,2%
set minute=%time_var:~3,2%
set second=%time_var:~6,2%
set formatted_time=%year%%month%%day%_%hour%%minute%%second%

:: 執行 git commit
git commit -m "auto_commit_%formatted_time%"
if errorlevel 1 (
    echo Failed to commit changes. Exiting...
    exit /b 1
)

pause
