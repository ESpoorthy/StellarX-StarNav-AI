# Tasks

## Task 1: Add recognition config section to config.yaml
- [x] 1.1 Add `recognition` section to config.yaml with keys: angle_tolerance_deg: 0.5, min_inliers: 3, confidence_success: 0.6, confidence_partial: 0.3, max_residual_deg: 1.0, ransac_iterations: 50
- [x] 1.2 Add `navigation` section to config.yaml with keys: min_correspondences: 2, max_residual_threshold_deg: 2.0, attitude_confidence_threshold: 0.3

## Task 2: Implement visualization functions
- [-] 2.1 Implement plot_star_field() in src/utils/visualization.py
- [-] 2.2 Implement plot_detections() in src/utils/visualization.py
- [-] 2.3 Implement plot_confidence_distribution() in src/utils/visualization.py
- [-] 2.4 Implement plot_attitude_residuals() in src/utils/visualization.py
- [-] 2.5 Add plot_recognition_result() function showing matched/unmatched/outlier stars

## Task 3: Implement catalog pattern matcher
- [-] 3.1 Implement match_pattern() in src/catalog/pattern_matcher.py using the recognition module

## Task 4: Create Phase 4 recognition tests
- [~] 4.1 Create tests/test_phase4_recognition.py with all 10 required Phase 4 tests

## Task 5: Create Phase 4 evaluation module
- [~] 5.1 Create src/evaluation/phase4_eval.py with run_phase4_evaluation() function

## Task 6: Create run_pipeline.py entry point
- [~] 6.1 Create run_pipeline.py at repo root

## Task 7: Create benchmark.py entry point
- [~] 7.1 Create benchmark.py at repo root

## Task 8: Verify Phase 1-3 tests still pass
- [~] 8.1 Run existing test suite and fix any regressions

## Task 9: Run Phase 4 tests and fix issues
- [~] 9.1 Run tests/test_phase4_recognition.py and fix all failures

## Task 10: Run Phase 4 evaluation
- [~] 10.1 Run Phase 4 evaluation and capture real metrics
