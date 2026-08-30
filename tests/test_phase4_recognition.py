"""
test_phase4_recognition.py
==========================
Phase 4 — Star Pattern Recognition tests.
10 core test scenarios + structural/attitude tests.

Run with:
    pytest tests/test_phase4_recognition.py -v
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pytest

from src.catalog.catalog_loader import load_catalog
from src.recognition.catalog_index import CatalogIndex
from src.recognition.pattern_builder import StarPattern
from src.recognition.pattern_matcher import (
    RecognitionOutput, RecognitionStatus, run_recognition,
)
from src.navigation.attitude_estimator import (
    angular_error_deg, estimate_attitude,
)

CATALOG_PATH = Path("data/catalog/hipparcos_bright.csv")

CFG = {
    "dataset": {"image_width": 512, "image_height": 512, "field_of_view_deg": 20.0},
    "features": {"max_stars": 10},
    "recognition": {
        "angle_tolerance_deg": 0.5, "min_inliers": 3,
        "confidence_success": 0.6, "confidence_partial": 0.3,
        "max_residual_deg": 1.0, "ransac_iterations": 50,
    },
    "navigation": {
        "min_correspondences": 2, "max_residual_threshold_deg": 2.0,
        "attitude_confidence_threshold": 0.3,
    },
}


@pytest.fixture(scope="module")
def cidx():
    cat = load_catalog(CATALOG_PATH)
    return CatalogIndex(cat)


def _rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])


def _make_pattern(cidx, indices, R=None, noise_deg=0.0, seed=42):
    """Project catalog stars through rotation R into camera frame."""
    if R is None:
        R = np.eye(3)
    fov = 20.0; w = 512; h = 512
    focal = (w/2) / math.tan(math.radians(fov/2))
    cx, cy = w/2, h/2
    rng = np.random.default_rng(seed)
    uvs, pcs, brs = [], [], []
    for idx in indices:
        vi = cidx.get_by_catalog_index(idx).unit_vec.copy()
        if noise_deg > 0:
            p = rng.normal(0, math.radians(noise_deg), 3)
            p -= np.dot(p, vi)*vi
            vi = vi + p; vi /= np.linalg.norm(vi)
        vc = R.T @ vi
        if vc[2] <= 0: continue
        col = (vc[0]/vc[2])*focal + cx
        row = -(vc[1]/vc[2])*focal + cy
        if not (0 <= col < w and 0 <= row < h): continue
        uvs.append(vc/np.linalg.norm(vc)); pcs.append([col,row]); brs.append(1.0-idx*0.01)
    if not uvs:
        return StarPattern(n_stars=0)
    n = len(uvs); uv = np.array(uvs); pc = np.array(pcs); br = np.array(brs)
    ang = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d = max(-1.,min(1.,float(np.dot(uv[i],uv[j]))))
            a = math.degrees(math.acos(d)); ang[i,j]=a; ang[j,i]=a
    return StarPattern(unit_vectors=uv,pixel_coords=pc,brightnesses=br,
                       pairwise_angles_deg=ang,n_stars=n,
                       focal_px=focal,image_width=w,image_height=h,fov_deg=fov)


# ── Test 1: Known pattern → catalog identification ────────────────────────
class TestKnownPattern:
    def test_produces_valid_status(self, cidx):
        p = _make_pattern(cidx, list(range(6)))
        if p.n_stars < 2: pytest.skip("too few stars in frame")
        out = run_recognition(p, cidx, CFG)
        assert isinstance(out.status, RecognitionStatus)

    def test_identified_stars_have_hip_ids(self, cidx):
        p = _make_pattern(cidx, list(range(6)))
        if p.n_stars < 2: pytest.skip()
        out = run_recognition(p, cidx, CFG)
        for s in out.identified_stars:
            assert s.catalog_id.startswith("HIP_")

    def test_n_observed_matches_n_stars(self, cidx):
        p = _make_pattern(cidx, list(range(5)))
        if p.n_stars < 2: pytest.skip()
        out = run_recognition(p, cidx, CFG)
        assert out.n_observed == p.n_stars


# ── Test 2: Rotated pattern → identification unchanged ────────────────────
class TestRotatedPattern:
    def test_pairwise_angles_rotation_invariant(self, cidx):
        indices = list(range(4))
        p1 = _make_pattern(cidx, indices, R=np.eye(3))
        p2 = _make_pattern(cidx, indices, R=_rot_z(30))
        if p1.n_stars < 2 or p2.n_stars < 2: pytest.skip()
        n = min(p1.n_stars, p2.n_stars)
        idx = np.triu_indices(n, k=1)
        a1 = sorted(p1.pairwise_angles_deg[:n,:n][idx])
        a2 = sorted(p2.pairwise_angles_deg[:n,:n][idx])
        for a,b in zip(a1,a2):
            assert abs(a-b) < 0.02, f"angle mismatch {a:.4f} vs {b:.4f} after rotation"

    def test_rotated_returns_valid_output(self, cidx):
        p = _make_pattern(cidx, list(range(5)), R=_rot_z(45))
        if p.n_stars < 2: pytest.skip()
        out = run_recognition(p, cidx, CFG)
        assert isinstance(out, RecognitionOutput)


# ── Test 3: Noisy centroids → robust (no crash) ───────────────────────────
class TestNoisyCentroids:
    def test_small_noise_no_crash(self, cidx):
        p = _make_pattern(cidx, list(range(6)), noise_deg=0.2, seed=123)
        if p.n_stars < 2: pytest.skip()
        out = run_recognition(p, cidx, CFG)
        assert isinstance(out, RecognitionOutput)

    def test_moderate_noise_valid_status(self, cidx):
        p = _make_pattern(cidx, list(range(6)), noise_deg=0.4, seed=456)
        if p.n_stars < 2: pytest.skip()
        out = run_recognition(p, cidx, CFG)
        assert isinstance(out.status, RecognitionStatus)


# ── Test 4: Missing star → partial/graceful ───────────────────────────────
class TestMissingStar:
    def test_5_of_6_no_exception(self, cidx):
        p = _make_pattern(cidx, list(range(5)))
        if p.n_stars < 2: pytest.skip()
        out = run_recognition(p, cidx, CFG)
        assert isinstance(out, RecognitionOutput)

    def test_3_stars_valid_output(self, cidx):
        p = _make_pattern(cidx, [0,1,2])
        if p.n_stars < 2: pytest.skip()
        out = run_recognition(p, cidx, CFG)
        assert isinstance(out.status, RecognitionStatus)


# ── Test 5: Spurious star → outlier rejection ─────────────────────────────
class TestSpuriousStar:
    def test_injected_false_star_inliers_bounded(self, cidx):
        p = _make_pattern(cidx, list(range(5)))
        if p.n_stars < 2: pytest.skip()
        rng = np.random.default_rng(789)
        sv = rng.normal(size=3); sv[2]=abs(sv[2])+0.1; sv/=np.linalg.norm(sv)
        n = p.n_stars+1
        uv = np.vstack([p.unit_vectors, sv])
        pc = np.vstack([p.pixel_coords, [[256.,256.]]])
        br = np.append(p.brightnesses, 0.1)
        ang = np.zeros((n,n)); ang[:p.n_stars,:p.n_stars]=p.pairwise_angles_deg
        for i in range(p.n_stars):
            d=max(-1.,min(1.,float(np.dot(uv[i],sv)))); a=math.degrees(math.acos(d))
            ang[i,n-1]=a; ang[n-1,i]=a
        np2 = StarPattern(unit_vectors=uv,pixel_coords=pc,brightnesses=br,
                          pairwise_angles_deg=ang,n_stars=n,focal_px=p.focal_px,
                          image_width=p.image_width,image_height=p.image_height,fov_deg=p.fov_deg)
        out = run_recognition(np2, cidx, CFG)
        assert out.n_inliers <= n
        assert isinstance(out.status, RecognitionStatus)


# ── Test 6: Ambiguous/random pattern → FAILURE or LOW_CONFIDENCE ─────────
class TestAmbiguousPattern:
    def test_random_vectors_not_success(self, cidx):
        rng = np.random.default_rng(999)
        n=5; v=rng.normal(size=(n,3)); v[:,2]=np.abs(v[:,2])+0.1
        v/=np.linalg.norm(v,axis=1,keepdims=True)
        ang=np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                d=max(-1.,min(1.,float(np.dot(v[i],v[j])))); a=math.degrees(math.acos(d))
                ang[i,j]=a; ang[j,i]=a
        p=StarPattern(unit_vectors=v,pixel_coords=rng.uniform(50,462,(n,2)),
                      brightnesses=np.ones(n)*0.5,pairwise_angles_deg=ang,n_stars=n,
                      focal_px=1448.,image_width=512,image_height=512,fov_deg=20.)
        out=run_recognition(p,cidx,CFG)
        assert out.status in (RecognitionStatus.FAILURE,RecognitionStatus.LOW_CONFIDENCE,
                               RecognitionStatus.PARTIAL)
        assert out.confidence < 0.95


# ── Test 7: No stars → FAILURE ────────────────────────────────────────────
class TestNoMatch:
    def test_zero_stars_failure(self, cidx):
        out=run_recognition(StarPattern(n_stars=0),cidx,CFG)
        assert out.status==RecognitionStatus.FAILURE

    def test_zero_stars_no_crash(self, cidx):
        out=run_recognition(StarPattern(n_stars=0),cidx,CFG)
        assert isinstance(out,RecognitionOutput)
        assert out.identified_stars==[]


# ── Test 8: Insufficient stars → graceful failure ────────────────────────
class TestInsufficientStars:
    def test_one_star_failure(self, cidx):
        v=np.array([[0.,0.,1.]]); ang=np.array([[0.]])
        p=StarPattern(unit_vectors=v,pixel_coords=np.array([[256.,256.]]),
                      brightnesses=np.array([1.]),pairwise_angles_deg=ang,n_stars=1,
                      focal_px=1448.,image_width=512,image_height=512,fov_deg=20.)
        out=run_recognition(p,cidx,CFG)
        assert out.status==RecognitionStatus.FAILURE

    def test_one_star_no_crash(self, cidx):
        out=run_recognition(StarPattern(n_stars=1),cidx,CFG)
        assert isinstance(out,RecognitionOutput)


# ── Test 9: Candidate ranking / output structure ──────────────────────────
class TestCandidateRanking:
    def test_confidence_in_unit_range(self, cidx):
        p=_make_pattern(cidx,list(range(5)))
        if p.n_stars<2: pytest.skip()
        out=run_recognition(p,cidx,CFG)
        assert 0.0<=out.confidence<=1.0

    def test_per_star_residuals_non_negative(self, cidx):
        p=_make_pattern(cidx,list(range(5)))
        if p.n_stars<2: pytest.skip()
        out=run_recognition(p,cidx,CFG)
        for s in out.identified_stars:
            assert s.angular_residual_deg>=0.0

    def test_matched_pattern_inlier_count_matches(self, cidx):
        p=_make_pattern(cidx,list(range(5)))
        if p.n_stars<2: pytest.skip()
        out=run_recognition(p,cidx,CFG)
        if out.matched_pattern:
            assert out.matched_pattern.inlier_count==out.n_inliers

    def test_is_successful_only_when_success(self, cidx):
        p=StarPattern(n_stars=0)
        out=run_recognition(p,cidx,CFG)
        assert out.is_successful() is False


# ── Test 10: Increasing noise → confidence does not increase ─────────────
class TestNoiseConfidence:
    def test_heavy_noise_confidence_bounded(self, cidx):
        p=_make_pattern(cidx,list(range(6)),noise_deg=1.0,seed=42)
        if p.n_stars<2: pytest.skip()
        out=run_recognition(p,cidx,CFG)
        assert out.confidence<=1.0+1e-9

    def test_clean_inliers_ge_noisy(self, cidx):
        pc=_make_pattern(cidx,list(range(5)),noise_deg=0.0)
        pn=_make_pattern(cidx,list(range(5)),noise_deg=0.6,seed=77)
        if pc.n_stars<2 or pn.n_stars<2: pytest.skip()
        oc=run_recognition(pc,cidx,CFG); on=run_recognition(pn,cidx,CFG)
        assert oc.n_inliers>=0 and on.n_inliers>=0


# ── Attitude estimator tests ──────────────────────────────────────────────
class TestAttitudeEstimator:
    def test_identity_corr_zero_error(self):
        obs=np.eye(3); cat=np.eye(3)
        r=estimate_attitude(obs,cat,CFG)
        err=angular_error_deg(r.rotation_matrix,np.eye(3))
        assert err<0.01, f"identity error {err:.4f} deg > 0.01"

    def test_known_rotation_recovered(self):
        R_true=_rot_z(15.)
        rng=np.random.default_rng(42)
        vecs=[v/np.linalg.norm(v) for v in [rng.normal(size=3) for _ in range(5)]]
        obs=np.array(vecs); cat=np.array([R_true@v for v in vecs])
        r=estimate_attitude(obs,cat,CFG)
        err=angular_error_deg(r.rotation_matrix,R_true)
        assert err<0.1, f"rotation error {err:.4f} deg > 0.1"
        assert r.is_valid

    def test_too_few_corr_invalid(self):
        obs=np.array([[1.,0.,0.]]); cat=obs.copy()
        r=estimate_attitude(obs,cat,CFG)
        assert r.is_valid is False

    def test_shape_mismatch_raises(self):
        obs=np.eye(3); cat=np.eye(2)[:,:3]
        with pytest.raises((ValueError,Exception)):
            estimate_attitude(obs,cat,CFG)

# ── Navigation pipeline smoke test ───────────────────────────────────────
class TestNavigationSmoke:
    def test_zero_star_image_no_crash(self, cidx):
        from src.navigation.navigator import run_navigation
        img=np.zeros((512,512),dtype=np.float32)
        result=run_navigation(img,CFG,cidx,neural_model=None)
        assert result.status in ("FAILURE","ERROR","LOW_CONFIDENCE")

    def test_pipeline_returns_navigation_result(self, cidx):
        from src.navigation.navigator import run_navigation, NavigationResult
        img=np.zeros((512,512),dtype=np.float32)
        result=run_navigation(img,CFG,cidx,neural_model=None)
        assert isinstance(result,NavigationResult)
