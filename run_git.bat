@echo off
cd /d "c:\Users\Sahithi\OneDrive\Desktop\StellarX-StarNav-AI"

echo === git pull --rebase ===
git pull --rebase origin main
echo pull exit: %ERRORLEVEL%

echo.
echo === git stash pop ===
git stash pop
echo stash pop exit: %ERRORLEVEL%

echo.
echo === git add . ===
git add .
echo add exit: %ERRORLEVEL%

echo.
echo === git status staged ===
git status

echo.
echo === git commit ===
git commit -m "feat: implement Phase 4 - star pattern recognition"
echo commit exit: %ERRORLEVEL%

echo.
echo === git push ===
git push
echo push exit: %ERRORLEVEL%
