########## module import ##########
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

# 프로젝트 구조에 맞게 임포트 경로를 확인하세요.
from .utiles import BaseExperiment, plotting 

########## model define ##########

class PINN(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key):
        # 입력 특징의 크기를 1로 지정합니다.
        self.net = eqx.nn.MLP(
            in_size=1,
            out_size=6, 
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.softplus, 
            key=key
        )
        
    def __call__(self, t):
        # [해결의 열쇠] 정적/동적 트레이서가 충돌하지 않도록 reshape를 절대 쓰지 않습니다.
        # 기하학적 차원 추가 없이, 오직 마지막 축에 차원 하나를 늘려주는 방식으로 
        # 스칼라든 벡터든 유연하게 받아내도록 처리합니다.
        return self.net(t[..., jnp.newaxis])

########## Experiment ##########
    
class Experiment(BaseExperiment):

    def __init__(self, ts, ys, ts_ge=None, width_size=64, depth=4, **kwargs):
        seed = kwargs.get("seed", 5678)
        self.lambda_p = kwargs.get("lambda_p", 1.0) 
        self.ge_multiplier = 2 

        model = PINN(
            width_size=width_size,
            depth=depth,
            key=jr.PRNGKey(seed),
        )

        super().__init__(model, ts, ys, **kwargs)

    def loss_fn(self, model, ts, ys):
        # 1. Data Loss 계산
        # ts를 완전한 스칼라 시퀀스로 풀어서 vmap을 매핑합니다.
        preds_data = jax.vmap(model)(ts) # (140, 6)
        I_pred = preds_data[:, 2]                 
        data_loss = jnp.mean(jnp.square(I_pred - ys) / jnp.max(ys).squeeze())

        # 2. Physics Loss 계산
        ts_ge = jnp.linspace(ts, ts[-1], len(ts) * self.ge_multiplier) # (280,)

        # 가상 그리드 전체에 대해 예측값 생성 (280, 6)
        vals = jax.vmap(model)(ts_ge) 
        S, E, I, R, bb, ss = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3], vals[:, 4], vals[:, 5]
        
        # [차원 폭발 방지 종결자]
        # jax.jacobian이나 jax.grad에 전체 모델을 바로 태우지 않고,
        # 입력 t(스칼라)를 받아 스칼라 배열을 뱉는 순수 매핑 함수들을 바깥에 정의합니다.
        def get_outputs(t):
            res = model(t)
            return res, res, res, res

        # 단일 시점 t에 대해 물리 상태 변수들의 미분값(그라디언트)을 구하는 함수를 감쌉니다.
        def compute_grads(t):
            # jax.jacobian을 사용해 각 컴포넌트별 스칼라 미분을 한 번에 계산
            jac = jax.jacobian(get_outputs)(t)
            return jac, jac, jac, jac

        # 최외곽에서 vmap을 적용해 (280,) 배열에 대한 미분 벡터들을 깔끔하게 추출합니다.
        # 이 정석 구조는 상위 scan 루프의 140 차원 간섭을 완벽히 차단합니다.
        dS_dt, dE_dt, dI_dt, dR_dt = jax.vmap(compute_grads)(ts_ge)
        
        # (280, 1) 또는 (280, 1, 1) 형태로 나온 미분 값들을 계산용 1D 벡터 (280,)로 스퀴즈합니다.
        dS_dt = dS_dt.squeeze()
        dE_dt = dE_dt.squeeze()
        dI_dt = dI_dt.squeeze()
        dR_dt = dR_dt.squeeze()

        N = S + E + I + R
        
        # SEIRS 시스템 고정 상수 정의
        mm, dd, r, kk, aa, gg = 0.0003671, 0.0027400, 0.0006762, 0.0001500, 0.0300000, 0.3500000

        # SEIRS 물리 방정식의 우변(RHS) 계산
        dS_rhs = - bb * I * S / N - mm * S + r * N + dd * R
        dE_rhs = bb * I * S / N - (mm + ss + kk) * E
        dI_rhs = ss * E - (mm + aa + gg) * I
        dR_rhs = kk * E + gg * I - mm * R - dd * R

        # 물리 잔차(Residual) 계산
        res_S = dS_dt - dS_rhs
        res_E = dE_dt - dE_rhs
        res_I = dI_dt - dI_rhs
        res_R = dR_dt - dR_rhs

        # 모든 잔차의 제곱 평균 계산
        physics_loss = jnp.mean(
            jnp.square(res_S) + jnp.square(res_E) + jnp.square(res_I) + jnp.square(res_R)
        )

        # 최종 가중합 Loss 반환
        total_loss = data_loss + self.lambda_p * physics_loss
        return total_loss
    
########## Evaluation ##########

def Evaluation(EX, ts_eval, loss_list):
    ts_data, ys_data, model = EX.ts, EX.ys, EX.model

    preds_eval = jax.vmap(model)(ts_eval) 

    ys_pred = preds_eval[:, :4]     
    beta_pred = preds_eval[:, 4]    
    sigma_pred = preds_eval[:, 5]   

    plotting(ts_data, ys_data, ts_eval, ys_pred, beta_pred, sigma_pred, loss_list)