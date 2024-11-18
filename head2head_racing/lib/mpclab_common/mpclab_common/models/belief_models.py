#!/usr/bin python3

import numpy as np
import scipy.linalg as la

import casadi as ca

import array
from typing import List, Tuple
import pdb

from mpclab_common.models.abstract_model import AbstractModel
from mpclab_common.models.dynamics_models import *
from mpclab_common.models.observation_models import *
from mpclab_common.models.model_types import BeliefConfig
from mpclab_common.pytypes import VehicleState, VehiclePrediction
from mpclab_common.track import get_track

class CasadiBeliefModel(AbstractModel):
    '''
    Base class for discrete time belief models formulated using casadi symbolics
    '''

    def __init__(self, model_config):
        super().__init__(model_config)

    def precompute_model(self):

        belief_args = [self.sym_b, self.sym_u]

        self.f_g = ca.Function('f_g', belief_args, [self.sym_g], self.options('f_g'))
        self.f_W = ca.Function('f_W', belief_args, [self.sym_W], self.options('f_W'))

        # Jacobians of g w.r.t. belief state and input
        self.sym_Db_g = ca.jacobian(self.sym_g, self.sym_b)
        self.sym_Du_g = ca.jacobian(self.sym_g, self.sym_u)

        self.f_Db_g = ca.Function('f_Db_g', belief_args, [self.sym_Db_g], self.options('f_Db_g'))
        self.f_Du_g = ca.Function('f_Du_g', belief_args, [self.sym_Du_g], self.options('f_Du_g'))

        # Jacobians of the columns of W w.r.t. belief state and input
        self.syms_Db_W = [ca.jacobian(self.sym_W[:,i], self.sym_b) for i in range(self.n_q)]
        self.syms_Du_W = [ca.jacobian(self.sym_W[:,i], self.sym_u) for i in range(self.n_q)]

        self.f_Db_W = ca.Function('f_Db_W', belief_args, self.syms_Db_W, self.options('f_Db_W'))
        self.f_Du_W = ca.Function('f_Du_W', belief_args, self.syms_Du_W, self.options('f_Du_W'))

        if self.code_gen and not self.jit:
            so_fns = [self.f_g, self.f_W, self.f_Db_g, self.f_Du_g, self.f_Db_W, self.f_Du_W]
            self.install_dir = self.build_shared_object(so_fns)

        return

    '''
    Nominal belief update
    '''
    def step(self, belief_state: VehicleState):
        b, u = self.state2bu(belief_state)
        args = [b, u]

        b_kp1 = np.array(self.f_g(*args)).squeeze()

        self.bu2state(belief_state, b_kp1, None)
        return

    def vec2tril(self, vec, n):
        M_cols = []
        idx_start = 0
        for i in range(n):
            M_cols.append(ca.vertcat(ca.DM.zeros(i), vec[idx_start:idx_start+(n-i)]))
            idx_start += (n-i)
        return ca.tril(ca.horzcat(*M_cols))

class CasadiSingleAgentBeliefModel(CasadiBeliefModel):
    '''
    Base class for discrete time belief models formulated using casadi symbolics
    '''

    def __init__(self, dynamics_model: CasadiDynamicsModel, observation_model: CasadiObservationModel, model_config: BeliefConfig = BeliefConfig()):
        super().__init__(model_config)

        self.dynamics_model = dynamics_model
        self.observation_model = observation_model

        self.use_mx = model_config.use_mx

        self.n_q = self.dynamics_model.n_q # State dimension
        self.n_u = self.dynamics_model.n_u # Input dimension
        self.n_m = self.dynamics_model.n_m # Process noise dimension
        self.n_n = self.observation_model.n_n # Measurement noise dimension

        self.n_b = self.n_q + self.n_q + int((self.n_q**2-self.n_q)/2) # Belief state dimension with triangular representation of covariance matrix

        self.tril_idxs = ca.Sparsity.lower(self.n_q).find()

        if self.use_mx:
            self.sym_b = ca.MX.sym('b', self.n_b) # Belief state
            self.sym_u = ca.MX.sym('u', self.n_u) # Input
            self.sym_m = ca.MX.sym('m', self.n_m) # Process noise
            self.sym_n = ca.MX.sym('n', self.n_n) # Measurement noise
            self.sym_x_hat = ca.vertcat(*ca.vertsplit(self.sym_b)[:self.n_q])
            S_vec = ca.vertcat(*ca.vertsplit(self.sym_b)[self.n_q:])
            self.sym_S = self.vec2tril(S_vec, self.n_q)
        else:
            self.sym_u = ca.SX.sym('u', self.n_u) # Input
            self.sym_m = ca.SX.sym('m', self.n_m) # Process noise
            self.sym_n = ca.SX.sym('n', self.n_n) # Measurement noise
            self.sym_x_hat = ca.SX.sym('x_hat', self.n_q) # Predicted state
            self.sym_S = ca.SX.sym('Sigma', ca.Sparsity.lower(self.n_q)) # Lower triangular of covariance of predicted state
            self.sym_b = ca.vertcat(self.sym_x_hat, self.sym_S[self.tril_idxs]) # Belief state
        # Note: the reason we use a lower triangular representation of the covariance matrix
        # is because casadi uses column-major indexing. So the order of elements in a numpy upper triangular matrix
        # obtained by np.triu_indices is equivalent to the order of elements in a casadi lower triangular matrix
        # obtained by Sparsity.find
        # pdb.set_trace()

        M = ca.SX.sym('M', ca.Sparsity.lower(self.n_q))
        chol = ca.Function('chol', [M], [ca.chol(ca.tril2symm(M))])

        dyn_args = [self.sym_x_hat, self.sym_u]

        # Jacobians of dynamics model w.r.t. state and process noise
        self.sym_A = self.dynamics_model.fAd(*dyn_args, ca.DM.zeros(self.n_m))
        self.sym_M = self.dynamics_model.fMd(*dyn_args, ca.DM.zeros(self.n_m))

        # Nominal state update
        self.sym_x_hat_p = self.dynamics_model.fd(*dyn_args, ca.DM.zeros(self.n_m))

        # Jacobians of observation model w.r.t. state and measurement noise
        self.sym_H = self.observation_model.fH(self.sym_x_hat_p, self.sym_u, ca.DM.zeros(self.n_n))
        self.sym_N = self.observation_model.fN(self.sym_x_hat_p, self.sym_u, ca.DM.zeros(self.n_n))

        # EKF update equations
        self.sym_G = ca.tril(self.sym_A @ ca.tril2symm(self.sym_S) @ ca.transpose(self.sym_A) + self.sym_M @ ca.transpose(self.sym_M))
        self.sym_K = ca.tril2symm(self.sym_G) @ ca.transpose(self.sym_H) @ ca.inv(self.sym_H @ ca.tril2symm(self.sym_G) @ ca.transpose(self.sym_H) + self.sym_N @ ca.transpose(self.sym_N))
        self.sym_P = ca.tril(self.sym_K @ self.sym_H @ ca.tril2symm(self.sym_G))

        # Stochastic belief model: b_k+1 = g(b_k, u_k) + W(b_k, u_k)\xi_k
        # where \xi_k is a unit Gaussian RV
        self.sym_g = ca.vertcat(self.sym_x_hat_p, (self.sym_G - self.sym_P)[self.tril_idxs])
        self.sym_W = ca.vertcat(chol(self.sym_P), ca.DM.zeros(len(self.tril_idxs), self.n_q))

        self.precompute_model()

        return

    def state2bu(self, belief_state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        x_hat, u = self.dynamics_model.state2qu(belief_state)
        if self.dynamics_model.curvature_model:
            S = np.array(belief_state.local_state_covariance)
        else:
            S = np.array(belief_state.global_state_covariance)

        b = np.concatenate((x_hat, S))
        return b, u

    def state2b(self, belief_state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        x_hat = self.dynamics_model.state2q(belief_state)
        if self.dynamics_model.curvature_model:
            S = np.array(belief_state.local_state_covariance)
        else:
            S = np.array(belief_state.global_state_covariance)

        b = np.concatenate((x_hat, S))
        return b

    def bu2state(self, belief_state: VehicleState, b: np.ndarray = None, u: np.ndarray = None):
        if b is not None:
            if u is not None:
                self.dynamics_model.qu2state(belief_state, b, u)
            else:
                self.dynamics_model.qu2state(belief_state, b, None)
            if self.dynamics_model.curvature_model:
                belief_state.local_state_covariance = b[self.n_q:]
            else:
                belief_state.global_state_covariance = b[self.n_q:]
        else:
            if u is not None:
                self.dynamics_model.qu2state(belief_state, None, u)
        return

    def bu2prediction(self, belief_prediction: VehiclePrediction, b: np.ndarray = None, u: np.ndarray = None):
        if b is not None:
            if u is not None:
                self.dynamics_model.qu2prediction(belief_prediction, b, u)
            else:
                self.dynamics_model.qu2prediction(belief_prediction, b, None)
            if self.dynamics_model.curvature_model:
                belief_prediction.local_state_covariance = b[:,self.n_q:].flatten()
            else:
                belief_prediction.global_state_covariance = b[:,self.n_q:].flatten()
        else:
            if u is not None:
                self.dynamics_model.qu2prediction(belief_prediction, None, u)
        return

class CasadiDecoupledMultiAgentBeliefModel(CasadiBeliefModel):
    '''
    Base class for discrete time belief models with decoupled dynamics and observations formulated using casadi symbolics
    '''

    def __init__(self, dynamics_models: List[CasadiDynamicsModel], observation_models: List[CasadiObservationModel], model_config: BeliefConfig = BeliefConfig()):
        super().__init__(model_config)

        self.dynamics_models = dynamics_models
        self.observation_models = observation_models

        self.use_mx = model_config.use_mx
        self.inter_agent_covariance = model_config.inter_agent_covariance

        self.M = len(self.dynamics_models) # Number of agents

        # Joint dimensions
        self.n_q, self.n_u, self.n_m, self.n_z, self.n_n = 0, 0, 0, 0, 0
        for i in range(self.M):
            self.n_q += self.dynamics_models[i].n_q # State dimension
            self.n_u += self.dynamics_models[i].n_u # Input dimension
            self.n_m += self.dynamics_models[i].n_m # Process noise dimension
            self.n_z += self.observation_models[i].n_z # Measurement dimension
            self.n_n += self.observation_models[i].n_n # Measurement noise dimension

        self.n_S = []
        # Joint belief state dimension
        if self.inter_agent_covariance:
            # Joint state dimension with triangular representation of covariance matrix including the cross covariances between agents
            self.n_S.append(self.n_q + int((self.n_q**2-self.n_q)/2))
        else:
            # Joint state dimension without the cross covariance between agents
            for i in range(self.M):
                self.n_S.append(self.dynamics_models[i].n_q + int((self.dynamics_models[i].n_q**2-self.dynamics_models[i].n_q)/2))
        self.n_b = self.n_q + np.sum(self.n_S)

        # Get the belief vector indices of elements of the joint covariance matrix corresponding to the covariance matrices for each agent
        self.agent_cov_idxs = []
        x_idx_start = 0
        b_idx_start = self.n_q
        joint_triu_idxs = np.array(np.triu_indices(self.n_q)).T
        for i in range(self.M):
            agent_triu_idxs = np.array(np.triu_indices(self.dynamics_models[i].n_q)).T + x_idx_start # Upper triangular indices for the agent covariance matrix in the joint covariance matrix
            agent_cov_idxs = []
            if self.inter_agent_covariance:
                for ai in agent_triu_idxs:
                    agent_cov_idxs.append(int(np.where(np.all(joint_triu_idxs == ai, axis=1))[0]+self.n_q))
            else:
                agent_cov_idxs = np.arange(agent_triu_idxs.shape[0]) + b_idx_start
                b_idx_start += agent_triu_idxs.shape[0]
            self.agent_cov_idxs.append(agent_cov_idxs)
            x_idx_start += self.dynamics_models[i].n_q

        # Helper function for creating block diagonal matrices
        def casadi_block_diag(block_mats, sparse=False, name='out'):
            dim1, dim2 = 0, 0
            for M in block_mats:
                dim1 += M.shape[0]
                dim2 += M.shape[1]
            if sparse:
                if self.use_mx:
                    sym_out = ca.MX.sym(name, ca.Sparsity(dim1, dim2))
                else:
                    sym_out = ca.SX.sym(name, ca.Sparsity(dim1, dim2))
            else:
                if self.use_mx:
                    sym_out = ca.MX.sym(name, dim1, dim2)
                else:
                    sym_out = ca.SX.sym(name, dim1, dim2)
            dim1_start, dim2_start = 0, 0
            for M in block_mats:
                sym_out[dim1_start:dim1_start+M.shape[0],dim2_start:dim2_start+M.shape[1]] = M
                dim1_start += M.shape[0]
                dim2_start += M.shape[1]
            return sym_out

        # Get the indices corresponding to lower triangular elements of a square matrix
        if self.inter_agent_covariance:
            self.tril_idxs = ca.Sparsity.lower(self.n_q).find()
        else:
            self.tril_idxs = []
            for i in range(self.M):
                self.tril_idxs.append(ca.Sparsity.lower(self.dynamics_models[i].n_q).find())

        if self.use_mx:
            # Define symbolic variables for joint vectors
            self.sym_b = ca.MX.sym('b', self.n_b) # Belief state
            self.sym_u = ca.MX.sym('u', self.n_u) # Input
            self.sym_m = ca.MX.sym('m', self.n_m) # Process noise
            self.sym_n = ca.MX.sym('n', self.n_n) # Measurement noise
            self.sym_x = ca.vertcat(*ca.vertsplit(self.sym_b)[:self.n_q])
            # Split into agent vectors
            sym_x, sym_u, sym_m, sym_n = [], [], [], []
            x_start, u_start, m_start, n_start = 0, 0, 0, 0
            for i in range(self.M):
                sym_x.append(ca.vertcat(*ca.vertsplit(self.sym_x)[x_start:x_start+self.dynamics_models[i].n_q]))
                sym_u.append(ca.vertcat(*ca.vertsplit(self.sym_u)[u_start:u_start+self.dynamics_models[i].n_u]))
                sym_m.append(ca.vertcat(*ca.vertsplit(self.sym_m)[m_start:m_start+self.dynamics_models[i].n_m]))
                sym_n.append(ca.vertcat(*ca.vertsplit(self.sym_n)[n_start:n_start+self.observation_models[i].n_n]))
                x_start += self.dynamics_models[i].n_q
                u_start += self.dynamics_models[i].n_u
                m_start += self.dynamics_models[i].n_m
                n_start += self.observation_models[i].n_n

            if self.inter_agent_covariance:
                S_vec = ca.vertcat(*ca.vertsplit(self.sym_b)[self.n_q:])
                self.sym_S = self.vec2tril(S_vec, self.n_q)
            else:
                self.sym_S = []
                s_start = self.n_q
                for i in range(self.M):
                    S_vec = ca.vertcat(*ca.vertsplit(self.sym_b)[s_start:s_start+self.n_S[i]])
                    self.sym_S.append(self.vec2tril(S_vec, self.dynamics_models[i].n_q))
                    s_start += self.n_S[i]
        else:
            # Define symbolic variables for each agent
            sym_u = [ca.SX.sym('u_%i' % i, self.dynamics_models[i].n_u) for i in range(self.M)] # Input
            sym_m = [ca.SX.sym('m_%i' % i, self.dynamics_models[i].n_m) for i in range(self.M)] # Process noise
            sym_n = [ca.SX.sym('n_%i' % i, self.observation_models[i].n_n) for i in range(self.M)] # Measurement noise
            sym_x = [ca.SX.sym('x_hat_%i' % i, self.dynamics_models[i].n_q) for i in range(self.M)] # Predicted state
            sym_S = [ca.SX.sym('S_%i' % i, ca.Sparsity.lower(self.dynamics_models[i].n_q)) for i in range(self.M)] # Lower triangular of covariance of predicted state
            # Concatenate into joint vector (we do this because slicing is inefficient in CasADi)
            self.sym_u = ca.vertcat(*sym_u)
            self.sym_m = ca.vertcat(*sym_m)
            self.sym_n = ca.vertcat(*sym_n)
            self.sym_x = ca.vertcat(*sym_x)
            if self.inter_agent_covariance:
                self.sym_S = ca.tril(casadi_block_diag(sym_S, name='S_joint'))
                self.sym_b = ca.vertcat(self.sym_x, self.sym_S[self.tril_idxs]) # Belief state
            else:
                self.sym_S = sym_S # Keep decoupled state covariance matrices
                S_vec = []
                for i, S in enumerate(sym_S):
                    S_vec.append(S[self.tril_idxs[i]])
                self.sym_b = ca.vertcat(self.sym_x, *S_vec) # Assemble belief vector
        # Note: the reason we use a lower triangular representation of the covariance matrix
        # is because casadi uses column-major indexing. So the order of elements in a numpy upper triangular matrix
        # obtained by np.triu_indices is equivalent to the order of elements in a casadi lower triangular matrix
        # obtained by Sparsity.find

        sym_A_i, sym_M_i, sym_H_i, sym_N_i, sym_x_p_i = [], [], [], [], []

        # Assemble vehicle dynamics input arguments for each agent
        for i, (dyn_mdl, obs_mdl) in enumerate(zip(self.dynamics_models, self.observation_models)):
            dyn_args_i = [sym_x[i], sym_u[i]]

            # Jacobians of dynamics model w.r.t. state and process noise
            sym_A_i.append(dyn_mdl.fAd(*dyn_args_i, ca.DM.zeros(dyn_mdl.n_m)))
            sym_M_i.append(dyn_mdl.fMd(*dyn_args_i, ca.DM.zeros(dyn_mdl.n_m)))

            # Nominal state update
            sym_x_p_i.append(dyn_mdl.fd(*dyn_args_i, ca.DM.zeros(dyn_mdl.n_m)))

            # Jacobians of observation model w.r.t. state and measurement noise
            sym_H_i.append(obs_mdl.fH(sym_x_p_i[-1], sym_u[i], ca.DM.zeros(obs_mdl.n_n)))
            sym_N_i.append(obs_mdl.fN(sym_x_p_i[-1], sym_u[i], ca.DM.zeros(obs_mdl.n_n)))

        # EKF update equations
        if self.inter_agent_covariance:
            M = ca.SX.sym('M', ca.Sparsity.lower(self.n_q))
            chol = ca.Function('chol', [M], [ca.chol(ca.tril2symm(M))])

            self.sym_A = casadi_block_diag(sym_A_i, sparse=True, name='A')
            self.sym_M = casadi_block_diag(sym_M_i, sparse=True, name='M')
            self.sym_H = casadi_block_diag(sym_H_i, sparse=True, name='H')
            self.sym_N = casadi_block_diag(sym_N_i, sparse=True, name='N')

            G = ca.tril(self.sym_A @ ca.tril2symm(self.sym_S) @ self.sym_A.T + self.sym_M @ self.sym_M.T)
            P = ca.tril(ca.tril2symm(G) @ self.sym_H.T @ ca.inv(self.sym_H @ ca.tril2symm(G) @ self.sym_H.T + self.sym_N @ self.sym_N.T) @ self.sym_H @ ca.tril2symm(G))

            # Stochastic belief model: b_k+1 = g(b_k, u_k) + W(b_k, u_k)\xi_k
            # where \xi_k is a unit Gaussian RV
            self.sym_g = ca.vertcat(*sym_x_p_i, (G - P)[self.tril_idxs])
            self.sym_W = ca.vertcat(chol(P), ca.DM.zeros(len(self.tril_idxs), self.n_q))
        else:
            self.sym_A = sym_A_i
            self.sym_M = sym_M_i
            self.sym_H = sym_H_i
            self.sym_N = sym_N_i
            g_var_i, W_i = [], []
            for i in range(self.M):
                M = ca.SX.sym('M', ca.Sparsity.lower(self.dynamics_models[i].n_q))
                chol = ca.Function('chol', [M], [ca.chol(ca.tril2symm(M))])

                G = ca.tril(self.sym_A[i] @ ca.tril2symm(self.sym_S[i]) @ ca.transpose(self.sym_A[i]) + self.sym_M[i] @ ca.transpose(self.sym_M[i]))
                P = ca.tril(ca.tril2symm(G) @ ca.transpose(self.sym_H[i]) @ ca.inv(self.sym_H[i] @ ca.tril2symm(G) @ ca.transpose(self.sym_H[i]) + self.sym_N[i] @ ca.transpose(self.sym_N[i])) @ self.sym_H[i] @ ca.tril2symm(G))

                g_var_i.append((G - P)[self.tril_idxs[i]])
                W_i.append(chol(P))

            self.sym_g = ca.vertcat(*sym_x_p_i, *g_var_i)
            self.sym_W = ca.vertcat(casadi_block_diag(W_i, sparse=True, name='W'),  ca.DM.zeros(np.sum(list(map(len,self.tril_idxs))), self.n_q))

        self.precompute_model()

        return

    '''
    Nominal belief update
    '''
    def step(self, belief_states: List[VehicleState]):
        b_joint, u_joint = self.state2bu(belief_states)

        args = [b_joint, u_joint]
        b_kp1 = np.array(self.f_g(*args)).squeeze()

        self.bu2state(belief_states, b_kp1, None)
        return

    def state2bu(self, belief_states: List[VehicleState]) -> Tuple[np.ndarray, np.ndarray]:
        b_joint = np.zeros(self.n_b)
        u_joint = np.zeros(self.n_u)
        x_idx_start = 0
        u_idx_start = 0
        for i in range(self.M):
            n_q = self.dynamics_models[i].n_q
            n_u = self.dynamics_models[i].n_u
            b_joint[x_idx_start:x_idx_start+n_q], u_joint[u_idx_start:u_idx_start+n_u] = self.dynamics_models[i].state2qu(belief_states[i])
            if self.dynamics_models[i].curvature_model:
                b_joint[self.agent_cov_idxs[i]] = np.array(belief_states[i].local_state_covariance)
            else:
                b_joint[self.agent_cov_idxs[i]] = np.array(belief_states[i].global_state_covariance)
            x_idx_start += n_q
            u_idx_start += n_u

        return b_joint, u_joint

    def state2b(self, belief_states: List[VehicleState]) -> np.ndarray:
        b_joint = np.zeros(self.n_b)
        x_idx_start = 0
        for i in range(self.M):
            n_q = self.dynamics_models[i].n_q
            b_joint[x_idx_start:x_idx_start+n_q] = self.dynamics_models[i].state2q(belief_states[i])
            if self.dynamics_models[i].curvature_model:
                b_joint[self.agent_cov_idxs[i]] = np.array(belief_states[i].local_state_covariance)
            else:
                b_joint[self.agent_cov_idxs[i]] = np.array(belief_states[i].global_state_covariance)
            x_idx_start += n_q

        return b_joint

    def bu2state(self, belief_states: List[VehicleState], b: np.ndarray = None, u: np.ndarray = None):
        x_idx_start = 0
        u_idx_start = 0
        for i in range(self.M):
            if b is not None:
                if u is not None:
                    self.dynamics_models[i].qu2state(belief_states[i], b[x_idx_start:x_idx_start+self.dynamics_models[i].n_q], u[u_idx_start:u_idx_start+self.dynamics_models[i].n_u])
                    u_idx_start += self.dynamics_models[i].n_u
                else:
                    self.dynamics_models[i].qu2state(belief_states[i], b[x_idx_start:x_idx_start+self.dynamics_models[i].n_q], None)
                if self.dynamics_models[i].curvature_model:
                    belief_states[i].local_state_covariance = b[self.agent_cov_idxs[i]].flatten()
                else:
                    belief_states[i].global_state_covariance = b[self.agent_cov_idxs[i]].flatten()
                x_idx_start += self.dynamics_models[i].n_q
            else:
                if u is not None:
                    self.dynamics_models[i].qu2state(belief_states[i], None, u[u_idx_start:u_idx_start+self.dynamics_models[i].n_u])
                    u_idx_start += self.dynamics_models[i].n_u
        return

    def bu2prediction(self, belief_predictions: List[VehiclePrediction], b: np.ndarray = None, u: np.ndarray = None):
        x_idx_start = 0
        u_idx_start = 0
        for i in range(self.M):
            n_q = self.dynamics_models[i].n_q
            n_u = self.dynamics_models[i].n_u
            if b is not None:
                if u is not None:
                    self.dynamics_models[i].qu2prediction(belief_predictions[i], b[:,x_idx_start:x_idx_start+n_q], u[:,u_idx_start:u_idx_start+n_u])
                    u_idx_start += n_u
                else:
                    self.dynamics_models[i].qu2prediction(belief_predictions[i], b[:,x_idx_start:x_idx_start+n_q], None)
                if self.dynamics_models[i].curvature_model:
                    belief_predictions[i].local_state_covariance = b[:,self.agent_cov_idxs[i]].flatten()
                else:
                    belief_predictions[i].global_state_covariance = b[:,self.agent_cov_idxs[i]].flatten()
                x_idx_start += n_q
            else:
                if u is not None:
                    self.dynamics_models[i].qu2prediction(belief_predictions[i], None, u[:,u_idx_start:u_idx_start+n_u])
                    u_idx_start += n_u
        return

def main():
    from mpclab_common.models.model_types import KinematicBicycleConfig, ObserverConfig
    from mpclab_common.models.dynamics_models import CasadiKinematicCLBicycle
    from mpclab_common.models.observation_models import CasadiKinematicBicycleCLFullStateObserver

    track_name = 'Lab_Track_barc'

    ego_dynamics_config =KinematicBicycleConfig(model_name='kinematic_bicycle_cl',
                                                track_name=track_name,
                                                dt=0.1,
                                                noise=True,
                                                noise_cov=4*np.eye(4),
                                                code_gen=False,
                                                jit=True,
                                                opt_flag='O3',
                                                verbose=True)
    ego_dynamics = CasadiKinematicCLBicycle(0.0, ego_dynamics_config)

    ego_observer_config = ObserverConfig(model_name='kinematic_bicycle_cl_full_state',
                                            noise=True,
                                            noise_cov=1*np.eye(4),
                                            code_gen=False,
                                            jit=True,
                                            opt_flag='O3',
                                            verbose=True)
    ego_observer = CasadiKinematicBicycleCLFullStateObserver(ego_observer_config)

    ego_belief_config = BeliefConfig(code_gen=False,
                                        jit=True,
                                        opt_flag='O3',
                                        verbose=True)
    ego_belief_model = CasadiSingleAgentBeliefModel(ego_dynamics, ego_observer, ego_belief_config)

    target_dynamics_config =KinematicBicycleConfig(model_name='kinematic_bicycle_cl',
                                                track_name=track_name,
                                                dt=0.1,
                                                noise=True,
                                                noise_cov=4*np.eye(4),
                                                code_gen=False,
                                                jit=True,
                                                opt_flag='O3',
                                                verbose=True)
    target_dynamics = CasadiKinematicCLBicycle(0.0, ego_dynamics_config)

    target_observer_config = ObserverConfig(model_name='kinematic_bicycle_cl_full_state',
                                            noise=True,
                                            noise_cov=1*np.eye(4),
                                            code_gen=False,
                                            jit=True,
                                            opt_flag='O3',
                                            verbose=True)
    target_observer = CasadiKinematicBicycleCLFullStateObserver(ego_observer_config)

    joint_belief_config = BeliefConfig(code_gen=False,
                                        jit=True,
                                        opt_flag='O3',
                                        verbose=True,
                                        use_mx=False,
                                        inter_agent_covariance=False)
    joint_belief_model = CasadiDecoupledMultiAgentBeliefModel([ego_dynamics, target_dynamics], [ego_observer, target_observer], joint_belief_config)
    #
    # # pdb.set_trace()
    # rng = np.random.default_rng()
    #
    # M_ego = rng.random((ego_dynamics.n_q, ego_dynamics.n_q))
    # S_ego = M_ego.dot(M_ego.T)
    # # print(S_ego)
    # ego_triu_idxs = np.triu_indices(ego_dynamics.n_q)
    # ego_belief_state = VehicleCoords(t=0.0, s=1.0, x_tran=0.2, e_psi=0.1, v_long=1.0, v_tran=0.0, psidot=0.0, u_a=1.0, u_steer=-0.1, local_state_covariance=array.array('d', S_ego[ego_triu_idxs]))
    #
    # M_target = rng.random((target_dynamics.n_q, target_dynamics.n_q))
    # S_target = M_target.dot(M_target.T)
    # # print(S_target)
    # target_triu_idxs = np.triu_indices(target_dynamics.n_q)
    # target_belief_state = VehicleCoords(t=0.0, s=2.0, x_tran=-0.2, e_psi=-0.1, v_long=0.5, v_tran=0.0, psidot=0.0, u_a=0.5, u_steer=0.5, local_state_covariance=array.array('d', S_target[target_triu_idxs]))
    #
    # # pdb.set_trace()
    # joint_belief = [ego_belief_state, target_belief_state]
    # joint_belief_model.step(joint_belief)
    pdb.set_trace()

if __name__ == '__main__':
    main()
