Set-Location "c:\Users\Sahithi\OneDrive\Desktop\StellarX-StarNav-AI"

Write-Output "=== git pull --rebase ==="
$pullResult = git pull --rebase origin main 2>&1
Write-Output $pullResult
Write-Output "pull exit: $LASTEXITCODE"

Write-Output ""
Write-Output "=== git stash pop ==="
$popResult = git stash pop 2>&1
Write-Output $popResult
Write-Output "stash pop exit: $LASTEXITCODE"

Write-Output ""
Write-Output "=== git add . ==="
$addResult = git add . 2>&1
Write-Output $addResult
Write-Output "add exit: $LASTEXITCODE"

Write-Output ""
Write-Output "=== git status (staged) ==="
$statusResult = git status 2>&1
Write-Output $statusResult

Write-Output ""
Write-Output "=== git commit ==="
$commitResult = git commit -m "feat: implement Phase 4 - star pattern recognition

- Add CatalogIndex (KD-tree, precomputed pairwise angles)
- Add PatternBuilder (pixel-to-unit-vector, angular patterns)
- Add PatternMatcher (vote accumulation, RANSAC, Wahba/SVD, confidence)
- Add AttitudeEstimator (Wahba/SVD, quaternion, Euler angles)
- Add NavigationPipeline (run_navigation, run_full_pipeline)
- Add OptimizedPipeline with cached catalog and tracemalloc benchmarking
- Implement visualization functions (plot_star_field, plot_detections,
  plot_confidence_distribution, plot_attitude_residuals, plot_recognition_result)
- Implement match_pattern() in catalog/pattern_matcher.py
- Add recognition and navigation sections to config.yaml
- Add tests/test_phase4_recognition.py (12 pass, 0 failures)
- Add src/evaluation/phase4_eval.py with real robustness metrics
- Add run_pipeline.py and benchmark.py CLI entry points
- Update docs/architecture.md with Phase 4 pipeline diagram" 2>&1
Write-Output $commitResult
Write-Output "commit exit: $LASTEXITCODE"

Write-Output ""
Write-Output "=== git push ==="
$pushResult = git push 2>&1
Write-Output $pushResult
Write-Output "push exit: $LASTEXITCODE"
