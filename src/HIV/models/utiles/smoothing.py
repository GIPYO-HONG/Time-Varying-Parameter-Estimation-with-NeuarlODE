"""
MSSB (Multi-Stage Smoothing-Based) approach for HIV parameter estimation
논문: Liang, Miao & Wu (2010), The Annals of Applied Statistics

Stage I  : Local polynomial smoothing → T, V, dT, dV, ddV 추정
Stage II : Pseudo-least squares (PsLS) → lambda, rho, c 추정  [논문 수식 (2.2)]
Stage III: Semiparametric B-spline regression → delta, N, eta(t) 추정

[수식 재유도 - 논문 (2.5) 수정]
V'  = N*delta*TI - c*V                          ...(1.3)
V'' = N*delta*(eta*TU*V - delta*TI) - c*V'
   → V'' + c*V' = N*delta*eta*TU*V - delta*(V'+c*V)

TU = T - (V'+cV)/(N*delta) 대입 →

Z = V''+c*V'
  = U1*eta(t) + U2*delta + U3*N*delta*eta(t)

where:
  U1(t) = -(V'*V + c*V^2)
  U2(t) = -(V' + c*V)
  U3(t) = T*V

※ 논문 (2.6)의 U1/U2 역할이 바뀌어 있음에 주의
  논문: Z = U1*delta + U2*eta + U3*N*delta*eta  (표기 불일치)
  실제: Z = U2*delta + U1*eta + U3*N*delta*eta
"""

import math
import numpy as np
from scipy.linalg import solve
from scipy.interpolate import BSpline


# ============================================================
# Stage I: Local Polynomial Smoothing
# ============================================================

def epanechnikov_kernel(u):
    """Epanechnikov kernel K(u) = 0.75*(1-u^2) for |u|<=1"""
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def local_poly_fit(t_eval, t_data, y_data, h, degree, deriv=0):
    """
    로컬 다항식 스무딩 - 논문 Section 2.1

    X^(q)(t) = xi_{q+1}^T (T_{1+q,t}^T W_t T_{1+q,t})^{-1} T_{1+q,t}^T W_t Y

    Parameters
    ----------
    t_eval : 추정할 시점 배열
    t_data : 관측 시점 배열
    y_data : 관측값 배열
    h      : bandwidth
    degree : 로컬 다항식 차수
    deriv  : 반환할 미분 차수 (0, 1, 2)
    """
    result = np.zeros(len(t_eval))

    for i, t0 in enumerate(t_eval):
        u = (t_data - t0) / h
        weights = epanechnikov_kernel(u)

        mask = weights > 0
        # bandwidth 내 점이 부족하면 2배 확장
        if mask.sum() < degree + 2:
            weights = epanechnikov_kernel(u * 0.5)
            mask = weights > 0

        t_loc = t_data[mask]
        y_loc = y_data[mask]
        w_loc = weights[mask]

        # 설계행렬: 각 열이 (t_i - t0)^j
        T_mat = np.column_stack([(t_loc - t0)**j for j in range(degree + 1)])
        W_mat = np.diag(w_loc)

        try:
            A = T_mat.T @ W_mat @ T_mat
            b = T_mat.T @ W_mat @ y_loc
            # 수치 안정성을 위한 작은 ridge
            coef = np.linalg.solve(A + 1e-12 * np.eye(degree + 1), b)
            # coef[j] = X^(j)(t0) / j!  →  X^(deriv)(t0) = coef[deriv] * deriv!
            result[i] = coef[deriv] * math.factorial(deriv)
        except np.linalg.LinAlgError:
            result[i] = np.nan

    return result


def stage1_smooth(t_data, T_data, V_data, h_T=None, h_V=None):
    """
    논문 Section 2.1: 로컬 다항식으로 T, V 및 미분 추정

    bandwidth: Silverman's rule → h = 1.06 * std(t) * n^(-1/5)
    - X(t)   : local linear    (degree=1, deriv=0)
    - X'(t)  : local quadratic (degree=2, deriv=1)
    - X''(t) : local cubic     (degree=3, deriv=2)
    """
    n = len(t_data)
    if h_T is None:
        h_T = 1.06 * np.std(t_data) * n**(-0.2)
    if h_V is None:
        h_V = 1.06 * np.std(t_data) * n**(-0.2)

    print(f"  Bandwidth: h_T={h_T:.4f}, h_V={h_V:.4f}")

    T_hat   = local_poly_fit(t_data, t_data, T_data, h_T, degree=1, deriv=0)
    dT_hat  = local_poly_fit(t_data, t_data, T_data, h_T, degree=2, deriv=1)

    V_hat   = local_poly_fit(t_data, t_data, V_data, h_V, degree=1, deriv=0)
    dV_hat  = local_poly_fit(t_data, t_data, V_data, h_V, degree=2, deriv=1)
    ddV_hat = local_poly_fit(t_data, t_data, V_data, h_V, degree=3, deriv=2)

    return {
        "t":   t_data,
        "T":   T_hat,
        "V":   V_hat,
        "dT":  dT_hat,
        "dV":  dV_hat,
        "ddV": ddV_hat,
    }


# ============================================================
# Stage II: Pseudo-Least Squares (PsLS)
# ============================================================
# 논문 수식 (2.2):
#   V'(t) = alpha0 + alpha1*T(t) + alpha2*T'(t) - c*V(t)
#
# 설계행렬: X = [1, T, dT, -V],  Y = dV
# 추정 후:  lambda = -alpha0/alpha2,  rho = alpha1/alpha2

def stage2_psls(stage1, ridge=1e-6):
    """논문 Section 2.2: lambda, rho, c 추정"""
    T  = stage1["T"]
    dT = stage1["dT"]
    V  = stage1["V"]
    dV = stage1["dV"]

    X = np.column_stack([np.ones_like(T), T, dT, -V])
    Y = dV

    beta = solve(
        X.T @ X + ridge * np.eye(4),
        X.T @ Y,
    )
    alpha0, alpha1, alpha2, c_hat = beta

    return {
        "lambda": -alpha0 / alpha2,
        "rho":    alpha1 / alpha2,
        "c":      c_hat,
    }


# ============================================================
# Stage III: Semiparametric B-spline Regression
# ============================================================
# 올바른 수식 (직접 유도):
#   Z(t)  = V''(t) + c*V'(t)
#   U1(t) = -(V'(t)*V(t) + c*V(t)^2)    ← eta 계수
#   U2(t) = -(V'(t) + c*V(t))            ← delta 계수
#   U3(t) = T(t)*V(t)                    ← N*delta*eta 계수
#
#   Z = U1*eta(t) + U2*delta + U3*N*delta*eta(t)
#
# η(t) ≈ sum_j a_j * B_{j,k}(t) 로 B-spline 근사하면
# → theta = [delta, a_1,...,a_s, (N*delta)*a_1,...,(N*delta)*a_s]
#   에 대한 선형 시스템

def make_bspline_basis(t, n_knots=5, degree=2):
    """
    B-spline basis matrix 생성

    논문: AICc로 order(=degree+1)와 knot 수 선택
    내부 knot: uniform 간격 (log-scale 버전은 select_bspline_params에서 옵션 제공)
    """
    t_min, t_max = t.min(), t.max()
    inner_knots = np.linspace(t_min, t_max, n_knots + 2)[1:-1]
    knots = np.concatenate([
        np.repeat(t_min, degree + 1),
        inner_knots,
        np.repeat(t_max, degree + 1),
    ])
    n_basis = len(inner_knots) + degree + 1
    basis = np.zeros((len(t), n_basis))
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        basis[:, j] = BSpline(knots, coef, degree)(t)
    return basis


def stage3_bspline(stage1, c_hat, n_knots=5, degree=2, ridge=1e-6):
    """
    논문 Section 2.3: B-spline 선형회귀로 delta, N, eta(t) 추정

    설계행렬:
      col 0       : U2          → delta
      col 1..s    : B[:,j]*U1   → a_j (eta B-spline 계수)
      col s+1..2s : B[:,j]*U3   → (N*delta)*a_j

    theta = [delta, a_1,...,a_s, N*delta*a_1,...,N*delta*a_s]
    """
    t   = stage1["t"]
    T   = stage1["T"]
    V   = stage1["V"]
    dV  = stage1["dV"]
    ddV = stage1["ddV"]

    Z  = ddV + c_hat * dV
    U1 = -(dV * V + c_hat * V**2)   # eta 계수
    U2 = -(dV + c_hat * V)           # delta 계수
    U3 = T * V                        # N*delta*eta 계수

    B = make_bspline_basis(t, n_knots=n_knots, degree=degree)
    s = B.shape[1]

    # 설계행렬 조립
    col_delta = U2.reshape(-1, 1)
    cols_a    = B * U1.reshape(-1, 1)    # B_j(t) * U1(t)
    cols_Nda  = B * U3.reshape(-1, 1)    # B_j(t) * U3(t)
    X_full = np.hstack([col_delta, cols_a, cols_Nda])

    theta = solve(
        X_full.T @ X_full + ridge * np.eye(X_full.shape[1]),
        X_full.T @ Z,
    )

    delta_hat = theta[0]
    a_hat     = theta[1 : 1 + s]
    Nda_hat   = theta[1 + s :]          # (N*delta)*a_j

    # N 역산: (N*delta*a_j) / (delta*a_j)
    denom = delta_hat * a_hat
    valid = np.abs(denom) > 1e-8 * (np.abs(denom).max() + 1e-30)
    if valid.sum() > 0:
        N_hat = np.median(Nda_hat[valid] / denom[valid])
    else:
        N_hat = np.nan

    # eta(t) 복원
    eta_hat = B @ a_hat

    return {
        "delta": delta_hat,
        "N":     N_hat,
        "eta":   eta_hat,
        "a":     a_hat,
        "residuals": Z - X_full @ theta,
    }


# ============================================================
# Model Selection: AICc
# ============================================================

def aicc(residuals, n_params, n_obs):
    rss = np.sum(residuals**2)
    if rss <= 0 or n_obs <= n_params + 1:
        return np.inf
    aic = n_obs * np.log(rss / n_obs) + 2 * n_params
    return aic + 2 * n_params * (n_params + 1) / (n_obs - n_params - 1)


def select_bspline_params(stage1, c_hat,
                          knot_range=range(3, 9),
                          degree_range=(2, 3),
                          ridge=1e-6):
    """AICc 기준으로 B-spline degree/n_knots 선택 - 논문 Table 2, 3"""
    t   = stage1["t"]
    T   = stage1["T"]
    V   = stage1["V"]
    dV  = stage1["dV"]
    ddV = stage1["ddV"]

    Z  = ddV + c_hat * dV
    U1 = -(dV * V + c_hat * V**2)
    U2 = -(dV + c_hat * V)
    U3 = T * V
    n_obs = len(t)

    best_aicc = np.inf
    best = {"degree": 2, "n_knots": 5}
    rows = []

    for degree in degree_range:
        for n_knots in knot_range:
            try:
                B = make_bspline_basis(t, n_knots=n_knots, degree=degree)
                s = B.shape[1]
                X = np.hstack([U2.reshape(-1,1),
                                B * U1.reshape(-1,1),
                                B * U3.reshape(-1,1)])
                theta = solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Z)
                resid = Z - X @ theta
                n_params = 1 + 2 * s
                val = aicc(resid, n_params, n_obs)
                rows.append((degree, n_knots, n_params, val))
                if val < best_aicc:
                    best_aicc = val
                    best = {"degree": degree, "n_knots": n_knots}
            except Exception:
                pass

    print("\n  [Model Selection - AICc]")
    print(f"  {'degree':>6} {'n_knots':>8} {'n_params':>9} {'AICc':>14}")
    for row in sorted(rows, key=lambda x: x[3]):
        marker = " ←" if (row[0] == best["degree"] and row[1] == best["n_knots"]) else ""
        print(f"  {row[0]:>6} {row[1]:>8} {row[2]:>9} {row[3]:>14.4f}{marker}")

    return best


# ============================================================
# MSSB 전체 파이프라인
# ============================================================

def mssb_estimate(t_data, T_data, V_data,
                  h_T=None, h_V=None,
                  auto_select=True,
                  n_knots=5, degree=2,
                  ridge=1e-6):
    """
    MSSB 전체 파이프라인 실행

    Parameters
    ----------
    t_data, T_data, V_data : 관측 데이터 (numpy array)
    h_T, h_V   : Stage I bandwidth (None이면 Silverman rule)
    auto_select: True면 AICc로 B-spline 파라미터 자동 선택
    n_knots    : auto_select=False일 때 사용할 knot 수
    degree     : auto_select=False일 때 사용할 B-spline degree
    ridge      : 수치 안정성을 위한 ridge 패널티
    """

    # --------------------------------------------------
    # Stage I
    # --------------------------------------------------
    print("\n--- Stage I: Local Polynomial Smoothing ---")
    stage1 = stage1_smooth(t_data, T_data, V_data, h_T=h_T, h_V=h_V)
    print(f"  dT  range: [{stage1['dT'].min():.4e},  {stage1['dT'].max():.4e}]")
    print(f"  dV  range: [{stage1['dV'].min():.4e},  {stage1['dV'].max():.4e}]")
    print(f"  ddV range: [{stage1['ddV'].min():.4e},  {stage1['ddV'].max():.4e}]")

    # --------------------------------------------------
    # Stage II
    # --------------------------------------------------
    print("\n--- Stage II: PsLS for lambda, rho, c ---")
    stage2 = stage2_psls(stage1, ridge=ridge)
    c_hat = stage2["c"]
    print(f"  lambda = {stage2['lambda']:.6f}")
    print(f"  rho    = {stage2['rho']:.6f}")
    print(f"  c      = {c_hat:.6f}")

    # --------------------------------------------------
    # Stage III
    # --------------------------------------------------
    print("\n--- Stage III: Semiparametric B-spline for delta, N, eta(t) ---")

    if auto_select:
        best = select_bspline_params(stage1, c_hat, ridge=ridge)
        n_knots = best["n_knots"]
        degree  = best["degree"]
        print(f"\n  선택된 B-spline: degree={degree}, n_knots={n_knots}")

    stage3 = stage3_bspline(
        stage1, c_hat,
        n_knots=n_knots, degree=degree,
        ridge=ridge,
    )

    if np.any(stage3["eta"] < 0):
        print(f"  Warning: eta(t) 음수 포함 (min={stage3['eta'].min():.4e})")

    # --------------------------------------------------
    # 결과 출력
    # --------------------------------------------------
    print("\n" + "=" * 45)
    print("FINAL RESULTS (MSSB)")
    print("=" * 45)
    print(f"  lambda = {stage2['lambda']:>12.4f}   (true: 36.0)")
    print(f"  rho    = {stage2['rho']:>12.6f}   (true: 0.108)")
    print(f"  c      = {c_hat:>12.4f}   (true: 3.0)")
    print(f"  delta  = {stage3['delta']:>12.6f}   (true: 0.5)")
    print(f"  N      = {stage3['N']:>12.4f}   (true: 1000.0)")
    print(f"\n  eta(t) 첫 10개:")
    for i, v in enumerate(stage3["eta"][:10]):
        t_i = t_data[i]
        print(f"    t={t_i:.2f}  eta={v:.6e}")

    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
    }


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    # data_generation.py 있으면 사용, 없으면 내부 시뮬레이션
    try:
        import jax.numpy as jnp
        from data_generation import get_data, eta

        y0 = jnp.array([600., 30., 10**5])
        t_jax = jnp.linspace(0., 20., 200)
        ys = get_data(t_jax, y0, eta)

        t_data = np.array(t_jax)
        T_data = np.array(ys[:, 0] + ys[:, 1])
        V_data = np.array(ys[:, 2])
        print("data_generation.py 사용")

    except ImportError:
        print("data_generation.py 없음 → 내부 시뮬레이션")
        from scipy.integrate import solve_ivp

        lam_true, rho_true = 36.0, 0.108
        N_true, delta_true, c_true = 1000.0, 0.5, 3.0

        def eta_true(t):
            return 9e-5 * (1 - 0.9 * np.cos(np.pi * t / 1000))

        def ode(t, y):
            TU, TI, V = y
            et = eta_true(t)
            return [
                lam_true - rho_true * TU - et * TU * V,
                et * TU * V - delta_true * TI,
                N_true * delta_true * TI - c_true * V,
            ]

        sol = solve_ivp(
            ode, [0, 20], [600., 30., 1e5],
            t_eval=np.linspace(0, 20, 200),
            method="RK45", rtol=1e-10, atol=1e-12,
        )
        t_data = sol.t
        T_data = sol.y[0] + sol.y[1]
        V_data = sol.y[2]

        # 논문 시뮬레이션 설정: sigma1=20, sigma2=100
        rng = np.random.default_rng(42)
        T_data = T_data + rng.normal(0, 20,  size=T_data.shape)
        V_data = V_data + rng.normal(0, 100, size=V_data.shape)

    results = mssb_estimate(
        t_data, T_data, V_data,
        auto_select=True,
        ridge=1e-6,
    )