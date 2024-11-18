#!/usr/bin python3
from typing import List

from cvxopt import spmatrix, matrix, solvers
from cvxopt.solvers import qp
from osqp import OSQP

import numpy as np
from numpy import linalg as la

from scipy import linalg
from scipy import sparse

from dataclasses import dataclass, field
import datetime
import pdb
import copy

from mpclab_common.pytypes import PythonMsg, VehicleState, OrientationEuler, Position

@dataclass
class MPCParams(PythonMsg):
    n: int = field(default=None) # dimension state space
    d: int = field(default=None) # dimension input space
    N: int = field(default=None) # horizon length

    A: np.array = field(default=None) # prediction matrices. Single matrix for LTI and list for LTV
    B: np.array = field(default=None) # prediction matrices. Single matrix for LTI and list for LTV

    Q: np.array = field(default=np.array((n, n))) # quadratic state cost
    R: np.array = field(default=None) # quadratic input cost
    Qf: np.array = field(default=None) # quadratic state cost final
    dR: np.array = field(default=None) # Quadratic rate cost
    
    Qslack: float = field(default=None) # it has to be a vector. Qslack = [linearSlackCost, quadraticSlackCost]
    Fx: np.array = field(default=None) # State constraint Fx * x <= bx
    bx: np.array = field(default=None)
    Fu: np.array = field(default=None) # State constraint Fu * u <= bu
    bu: np.array = field(default=None)
    xRef: np.array = field(default=None)

    slacks: bool = field(default=True)
    timeVarying: bool = field(default=False)

    def __post_init__(self):
        if self.Qf is None: self.Qf = np.zeros((self.n, self.n))
        if self.dR is None: self.dR = np.zeros(self.d)
        if self.xRef is None: self.xRef = np.zeros(self.n)

############################################################################################
####################################### MPC CLASS ##########################################
############################################################################################
class MPC():
    """Model Predicitve Controller class
    Methods (needed by user):
        solve: given system's state xt compute control action at
    Arguments:
        mpcParameters: model paramters
    """
    def __init__(self,  mpcParameters, predictiveModel=[], print_method=print):
        """Initialization
        Arguments:
            mpcParameters: struct containing MPC parameters
        """
        self.N      = mpcParameters.N
        self.Qslack = mpcParameters.Qslack
        self.Q      = mpcParameters.Q
        self.Qf     = mpcParameters.Qf
        self.R      = mpcParameters.R
        self.dR     = mpcParameters.dR
        self.n      = mpcParameters.n
        self.d      = mpcParameters.d
        self.A      = mpcParameters.A
        self.B      = mpcParameters.B
        self.Fx     = mpcParameters.Fx
        self.Fu     = mpcParameters.Fu
        self.bx     = mpcParameters.bx
        self.bu     = mpcParameters.bu
        self.xRef   = mpcParameters.xRef
        # print(self.xRef)

        self.slacks          = mpcParameters.slacks
        self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel

        if self.timeVarying == True:
            self.xLin = self.predictiveModel.xStored[-1][0:self.N+1,:]
            self.uLin = self.predictiveModel.uStored[-1][0:self.N,:]
            self.computeLTVdynamics()

        self.OldInput = np.zeros((1,2)) # TO DO fix size

        # Build matrices for inequality constraints
        self.buildIneqConstr()
        self.buildCost()
        self.buildEqConstr()

        self.xPred = []

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0

        self.print_method = print_method

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state
        """
        # If LTV active --> identify system model
        sysid_time, solver_update_time = 0, 0
        if self.timeVarying == True:
            t_s = datetime.datetime.now()
            self.computeLTVdynamics()
            sysid_time = (datetime.datetime.now() - t_s).total_seconds()

            t_s = datetime.datetime.now()
            self.buildCost()
            self.buildEqConstr()
            solver_update_time = (datetime.datetime.now() - t_s).total_seconds()

        self.addTerminalComponents(x0)

        # Solve QP
        t_s = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x0),self.L_FTOCP))
        if self.feasible:
            self.unpackSolution()
            self.feasibleStateInput()
        solve_time = (datetime.datetime.now() - t_s).total_seconds()

        # self.print_method('sys ID time: %g, solver update time: %g, solve time: %g' % (sysid_time, solver_update_time, solve_time))

        # If LTV active --> compute state-input linearization trajectory
        if self.timeVarying == True:
            self.xLin = np.vstack((self.xPred[1:, :], self.zt))
            self.uLin = np.vstack((self.uPred[1:, :], self.zt_u))

        # update applied input
        self.OldInput = self.uPred[0,:]
        self.timeStep += 1

    def computeLTVdynamics(self):
        # Estimate system dynamics
        self.A = []; self.B = []; self.C =[]
        for i in range(0, self.N):
            Ai, Bi, Ci = self.predictiveModel.regressionAndLinearization(self.xLin[i], self.uLin[i])
            self.A.append(Ai); self.B.append(Bi); self.C.append(Ci)

    def addTerminalComponents(self, x0):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def feasibleStateInput(self):
        self.zt   = self.xPred[-1,:]
        self.zt_u = self.uPred[-1,:]

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.n*(self.N+1))]),(self.N+1,self.n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.n*(self.N+1)+np.arange(self.d*self.N)]),(self.N, self.d)))).T
        
    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        rep_a = [self.Fx] * (self.N)
        Mat = linalg.block_diag(*rep_a)
        NoTerminalConstr = np.zeros((np.shape(Mat)[0], self.n))  # The last state is unconstrained. There is a specific function add the terminal constraints (so that more complicated terminal constrains can be handled)
        Fxtot = np.hstack((Mat, NoTerminalConstr))
        bxtot = np.tile(np.squeeze(self.bx), self.N)

        # Let's start by computing the submatrix of F relates with the input
        rep_b = [self.Fu] * (self.N)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.N)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need
        if self.slacks == True:
            nc_x = self.Fx.shape[0] # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x*self.N))
            addSlack[0:nc_x*(self.N), 0:nc_x*(self.N)] = -np.eye(nc_x*(self.N))
            # Now constraint slacks >= 0
            I = - np.eye(nc_x*self.N); Zeros = np.zeros((nc_x*self.N, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x*self.N)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))

    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.n * (self.N + 1))
        Gu = np.zeros((self.n * (self.N + 1), self.d * (self.N)))

        E = np.zeros((self.n * (self.N + 1), self.n))
        E[np.arange(self.n)] = np.eye(self.n)

        L = np.zeros(self.n * (self.N + 1))

        for i in range(0, self.N):
            if self.timeVarying == True:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A[i]
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B[i]
                L[(self.n + i*self.n):(self.n + i*self.n + self.n)]                                  =  self.C[i]
            else:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B

        if self.slacks == True:
            self.G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], self.Fx.shape[0]*self.N) ) ) ) 
        else:
            self.G = np.hstack((Gx, Gu))
    
        self.E = E
        self.L = L

    def buildCost(self):
        # The cost is: (1/2) * z' H z + q' z
        listQ = [self.Q] * (self.N)
        Hx = linalg.block_diag(*listQ)

        listTotR = [self.R + 2 * np.diag(self.dR)] * (self.N) # Need to add dR for the derivative input cost
        Hu = linalg.block_diag(*listTotR)
        # Need to condider that the last input appears just once in the difference
        for i in range(0, self.d):
            Hu[ i - self.d, i - self.d] = Hu[ i - self.d, i - self.d] - self.dR[i]

        # Derivative Input Cost
        OffDiaf = -np.tile(self.dR, self.N-1)
        np.fill_diagonal(Hu[self.d:], OffDiaf)
        np.fill_diagonal(Hu[:, self.d:], OffDiaf)
        
        # Cost linear term for state and input
        q = - 2 * np.dot(np.append(np.tile(self.xRef, self.N + 1), np.zeros(self.R.shape[0] * self.N)), linalg.block_diag(Hx, self.Qf, Hu))
        # Derivative Input (need to consider input at previous time step)
        q[self.n*(self.N+1):self.n*(self.N+1)+self.d] = -2 * np.dot( self.OldInput, np.diag(self.dR) )
        if self.slacks == True: 
            quadSlack = self.Qslack[0] * np.eye(self.Fx.shape[0]*self.N)
            linSlack  = self.Qslack[1] * np.ones(self.Fx.shape[0]*self.N )
            self.H = linalg.block_diag(Hx, self.Qf, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else: 
            self.H = linalg.block_diag(Hx, self.Qf, Hu)
            self.q = q 
 
        self.H = 2 * self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """ 
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """  
        self.osqp = OSQP()
        qp_A = sparse.vstack([G, A]).tocsc()
        l = -np.inf * np.ones(len(h))
        qp_l = np.hstack([l, b])
        qp_u = np.hstack([h, b])

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0
            self.print_method(res.info.status)
        self.Solution = res.x

############## Below LMPC class which is a child of the MPC super class
class LMPC(MPC):
    """Create the LMPC
    Methods (needed by user):
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
    """
    def __init__(self, numSS_Points, numSS_it, QterminalSlack, mpcPrameters, predictiveModel, dt = 0.1, print_method=print):
        """Initialization
        Arguments:
            numSS_Points: number of points selected from the previous trajectories to build SS
            numSS_it: number of previois trajectories selected to build SS
            N: horizon length
            Q,R: weight to define cost function h(x,u) = ||x||_Q + ||u||_R
            dR: weight to define the input rate cost h(x,u) = ||x_{k+1}-x_k||_dR
            n,d: state and input dimensiton
            shift: given the closest point x_t^j to x(t) the controller start selecting the point for SS from x_{t+shift}^j
            map: map
            Laps: maximum number of laps the controller can run (used to avoid dynamic allocation)
            TimeLMPC: maximum time [s] that an lap can last (used to avoid dynamic allocation)
            Solver: solver used in the reformulation of the LMPC as QP
        """
        super().__init__(mpcPrameters, predictiveModel, print_method=print_method)
        self.numSS_Points = numSS_Points
        self.numSS_it     = numSS_it
        self.QterminalSlack = QterminalSlack

        self.OldInput = np.zeros((1,2))
        self.xPred    = []

        # Initialize the following quantities to avoid dynamic allocation
        self.LapTime = []        # Time at which each j-th iteration is completed
        self.SS         = []    # Sampled Safe SS
        self.uSS        = []    # Input associated with the points in SS
        self.Qfun       = []       # Qfun: cost-to-go from each point in SS
        self.SS_glob    = []   # SS in global (X-Y) used for plotting

        self.xStoredPredTraj     = []
        self.xStoredPredTraj_it  = []
        self.uStoredPredTraj     = []
        self.uStoredPredTraj_it  = []
        self.SSStoredPredTraj    = []
        self.SSStoredPredTraj_it = []

        self.zt = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])

        # Initialize the controller iteration
        self.it      = 0

        # Build matrices for inequality constraints
        self.buildIneqConstr()
        self.buildCost()
        self.addSafeSetIneqConstr()

    def addSafeSetIneqConstr(self):
        # Add positiviti constraints for lambda_{SafeSet}. Note that no constraint is enforced on slack_{SafeSet} ---> add np.hstack(-np.eye(self.numSS_Points), np.zeros(self.n)) 
        self.F_FTOCP = sparse.csc_matrix( linalg.block_diag( self.F, np.hstack((-np.eye(self.numSS_Points), np.zeros((self.numSS_Points, self.n)))) ) )
        self.b_FTOCP = np.append(self.b, np.zeros(self.numSS_Points))
    
    def addSafeSetEqConstr(self):
        # Add constrains for x, u, slack
        xTermCons = np.zeros((self.n, self.G.shape[1]))
        xTermCons[:, self.N * self.n:(self.N + 1) * self.n] = np.eye(self.n)
        G_x_u_slack = np.vstack((self.G, xTermCons))
        # Constraint for lambda_{SaFeSet, slack_{safeset}} to enforce safe set
        G_lambda_slackSafeSet = np.vstack( (np.zeros((self.G.shape[0], self.SS_PointSelectedTot.shape[1] + self.n)), np.hstack((-self.SS_PointSelectedTot, np.eye(self.n)))) )
        # Constraints on lambda = 1
        G_lambda = np.append(np.append(np.zeros(self.G.shape[1]), np.ones(self.SS_PointSelectedTot.shape[1])), np.zeros(self.n))
        # Put all together
        self.G_FTOCP = sparse.csc_matrix(np.vstack((np.hstack((G_x_u_slack, G_lambda_slackSafeSet)), G_lambda)))
        self.E_FTOCP = np.vstack((self.E, np.zeros((self.n+1,self.n)))) # adding n for terminal constraint and 1 for lambda = 1
        self.L_FTOCP = np.append(np.append(self.L, np.zeros(self.n)), 1)

    def addSafeSetCost(self):
        # need to multiply the quadratic term as cost is (1/2) z'*Q*z
        self.H_FTOCP = sparse.csc_matrix(linalg.block_diag(self.H, np.zeros((self.SS_PointSelectedTot.shape[1], self.SS_PointSelectedTot.shape[1])), 2*self.QterminalSlack) )
        self.q_FTOCP = np.append(np.append(self.q, self.Qfun_SelectedTot), np.zeros(self.n))
   
    def unpackSolution(self):
        stateIdx = self.n*(self.N+1)
        inputIdx = stateIdx + self.d*self.N
        slackIdx = inputIdx + self.Fx.shape[0]*self.N
        lambdIdx = slackIdx + self.SS_PointSelectedTot.shape[1]
        sTermIdx = lambdIdx + self.n

        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.n*(self.N+1))]),(self.N+1,self.n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.n*(self.N+1)+np.arange(self.d*self.N)]),(self.N, self.d)))).T
        self.slack = self.Solution[inputIdx:slackIdx]
        self.lambd = self.Solution[slackIdx:lambdIdx]
        self.slackTerminal = self.Solution[lambdIdx:]

        self.xStoredPredTraj_it.append(self.xPred)
        self.uStoredPredTraj_it.append(self.uPred)
        self.SSStoredPredTraj_it.append(self.SS_PointSelectedTot.T)
        

    def feasibleStateInput(self):
        self.zt = np.dot(self.Succ_SS_PointSelectedTot, self.lambd)
        self.zt_u = np.dot(self.Succ_uSS_PointSelectedTot, self.lambd)

    def addTerminalComponents(self,x0):
        """add terminal constraint and terminal cost
        Arguments:
            x: initial condition
        """        
        # Update zt and xLin is they have crossed the finish line. We want s \in [0, track_length]
        if (self.zt[4]-x0[4] > self.predictiveModel.map.track_length/2):
            self.zt[4] = np.max([self.zt[4] - self.predictiveModel.map.track_length,0])
            self.xLin[4,-1] = self.xLin[4,-1]- self.predictiveModel.map.track_length
        sortedLapTime = np.argsort(np.array(self.LapTime))

        # Select Points from historical data. These points will be used to construct the terminal cost function and constraint set
        SS_PointSelectedTot = np.empty((self.n, 0))
        Succ_SS_PointSelectedTot = np.empty((self.n, 0))
        Succ_uSS_PointSelectedTot = np.empty((self.d, 0))
        Qfun_SelectedTot = np.empty((0))
        for jj in sortedLapTime[0:self.numSS_it]:
            SS_PointSelected, uSS_PointSelected, Qfun_Selected = self.selectPoints(jj, self.zt, self.numSS_Points / self.numSS_it + 1)
            Succ_SS_PointSelectedTot =  np.append(Succ_SS_PointSelectedTot, SS_PointSelected[:,1:], axis=1)
            Succ_uSS_PointSelectedTot =  np.append(Succ_uSS_PointSelectedTot, uSS_PointSelected[:,1:], axis=1)
            SS_PointSelectedTot      = np.append(SS_PointSelectedTot, SS_PointSelected[:,0:-1], axis=1)
            Qfun_SelectedTot         = np.append(Qfun_SelectedTot, Qfun_Selected[0:-1], axis=0)

        self.Succ_SS_PointSelectedTot = Succ_SS_PointSelectedTot
        self.Succ_uSS_PointSelectedTot = Succ_uSS_PointSelectedTot
        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot = Qfun_SelectedTot
        
        # Update terminal set and cost
        self.addSafeSetEqConstr()
        self.addSafeSetCost()

    def addTrajectory(self, x, u, x_glob):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            x: closed-loop trajectory
            u: applied inputs
            x_gloab: closed-loop trajectory in global frame
        """
        self.LapTime.append(x.shape[0])
        self.SS.append(x)
        self.SS_glob.append(x_glob)
        self.uSS.append(u)
        self.Qfun.append(self.computeCost(x,u))

        if self.it == 0:
            self.xLin = self.SS[self.it][1:self.N + 2, :]
            self.uLin = self.uSS[self.it][1:self.N + 1, :]

        self.xStoredPredTraj.append(self.xStoredPredTraj_it)
        self.xStoredPredTraj_it = []

        self.uStoredPredTraj.append(self.uStoredPredTraj_it)
        self.uStoredPredTraj_it = []

        self.SSStoredPredTraj.append(self.SSStoredPredTraj_it)
        self.SSStoredPredTraj_it = []

        self.it = self.it + 1
        self.timeStep = 0

    def computeCost(self, x, u):
        """compute roll-out cost
        Arguments:
            x: closed-loop trajectory
            u: applied inputs
        """
        Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
        # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
        # We start from the last element of the vector x and we sum the running cost
        for i in range(0, x.shape[0]):
            if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
                Cost[x.shape[0] - 1 - i] = 0
            elif x[x.shape[0] - 1 - i, 4]< self.predictiveModel.map.track_length:
                Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
            else:
                Cost[x.shape[0] - 1 - i] = 0

        return Cost

    def addPoint(self, x, u):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
        """
        self.SS[self.it - 1]  = np.append(self.SS[self.it - 1], np.array([x + np.array([0, 0, 0, 0, self.predictiveModel.map.track_length, 0])]), axis=0)
        self.uSS[self.it - 1] = np.append(self.uSS[self.it - 1], np.array([u]),axis=0)
        self.Qfun[self.it - 1] = np.append(self.Qfun[self.it - 1], self.Qfun[self.it - 1][-1]-1)
        # The above two lines are needed as the once the predicted trajectory has crossed the finish line the goal is
        # to reach the end of the lap which is about to start

    def selectPoints(self, it, zt, numPoints):
        """selecte (numPoints)-nearest neivbor to zt. These states will be used to construct the safe set and the value function approximation
        Arguments:
            x: current state
            u: current input
        """
        x = self.SS[it]
        u = self.uSS[it]
        oneVec = np.ones((x.shape[0], 1))
        x0Vec = (np.dot(np.array([zt]).T, oneVec.T)).T
        diff = x - x0Vec
        norm = la.norm(diff, 1, axis=1)
        MinNorm = np.argmin(norm)

        if (MinNorm >= numPoints/2) and (x.shape[0] - MinNorm >= numPoints/2):
            indexSSandQfun = range(MinNorm - int(numPoints/2), MinNorm + int(numPoints/2) + 1)
        elif (MinNorm < numPoints/2):
            indexSSandQfun = range(MinNorm, MinNorm + int(numPoints))
        elif (x.shape[0] - MinNorm < numPoints/2):
            indexSSandQfun = range(MinNorm - int(numPoints), MinNorm)

        SS_Points  = x[indexSSandQfun, :].T
        SSu_Points = u[indexSSandQfun, :].T
        Sel_Qfun = self.Qfun[it][indexSSandQfun]

        # Modify the cost if the predicion has crossed the finisch line
        if self.xPred == []:
            Sel_Qfun = self.Qfun[it][indexSSandQfun]
        elif (np.all((self.xPred[:, 4] > self.predictiveModel.map.track_length) == False)):
            Sel_Qfun = self.Qfun[it][indexSSandQfun]
        elif it < self.it - 1:
            Sel_Qfun = self.Qfun[it][indexSSandQfun] + self.Qfun[it][0]
        else:
            sPred = self.xPred[:, 4]
            predCurrLap = self.N - sum(sPred > self.predictiveModel.map.track_length)
            currLapTime = self.timeStep
            Sel_Qfun = self.Qfun[it][indexSSandQfun] + currLapTime + predCurrLap

        return SS_Points, SSu_Points, Sel_Qfun

# This class is not generic and is tailored to the autonomous racing problem.
# The only method need the LT-MPC and the LMPC is regressionAndLinearization, which given a state-action pair
# compute the matrices A,B,C such that x_{k+1} = A x_k + Bu_k + C

class PredictiveModel():
    def __init__(self,  n, d, map, trToUse):
        self.map = map
        self.n = n # state dimension
        self.d = d # input dimention
        self.xStored = []
        self.uStored = []
        self.MaxNumPoint = 7 # max number of point per lap to use 
        self.h = 5 # bandwidth of the Kernel for local linear regression
        self.lamb = 0.0 # regularization
        # self.lamb = 1e-3 # regularization
        self.dt = 0.1
        self.scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0]])

        self.stateFeatures    = [0, 1, 2]
        self.inputFeaturesVx  = [1]
        self.inputFeaturesLat = [0]
        self.usedIt = [i for i in range(trToUse)]
        self.lapTime = []
    

    def addTrajectory(self, x, u):
        if self.lapTime == [] or x.shape[0] >= self.lapTime[-1]:
            self.xStored.append(x)
            self.uStored.append(u)
            self.lapTime.append(x.shape[0])
        else:
            for i in range(0, len(self.xStored)):
                if x.shape[0] < self.lapTime[i]:
                    self.xStored.insert(i, x) 
                    self.uStored.insert(i, u) 
                    self.lapTime.insert(i, x.shape[0]) 
                    break

    def regressionAndLinearization(self, x, u):
        Ai = np.zeros((self.n, self.n))
        Bi = np.zeros((self.n, self.d))
        Ci = np.zeros(self.n)

        # Compute Index to use for each stored lap
        xuLin = np.hstack((x[self.stateFeatures], u[:]))
        self.indexSelected = []
        self.K = []
        for ii in self.usedIt:
            indexSelected_i, K_i = self.computeIndices(xuLin, ii)
            self.indexSelected.append(indexSelected_i)
            self.K.append(K_i)
        # self.print_method("xuLin: ",xuLin)
        # self.print_method("aaa indexSelected: ", self.indexSelected)

        # =========================
        # ====== Identify vx ======
        Q_vx, M_vx = self.compute_Q_M(self.inputFeaturesVx, self.usedIt)

        yIndex = 0
        b_vx = self.compute_b(yIndex, self.usedIt, M_vx)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesVx], Ci[yIndex] = self.LMPC_LocLinReg(Q_vx, b_vx, self.inputFeaturesVx)

        # =======================================
        # ====== Identify Lateral Dynamics ======
        Q_lat, M_lat = self.compute_Q_M(self.inputFeaturesLat, self.usedIt)

        yIndex = 1  # vy
        b_vy = self.compute_b(yIndex, self.usedIt, M_lat)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_vy, self.inputFeaturesLat)

        yIndex = 2  # wz
        b_wz = self.compute_b(yIndex, self.usedIt, M_lat)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_wz, self.inputFeaturesLat)

        # ===========================
        # ===== Linearization =======
        vx = x[0]; vy   = x[1]
        wz = x[2]; epsi = x[3]
        s  = x[4]; ey   = x[5]
        dt = self.dt

        if s < 0:
            self.print_method("s is negative, here the state: \n", x)

        startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
        cur = self.map.get_curvature(s)
        den = 1 - cur * ey

        # ===========================
        # ===== Linearize epsi ======
        # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
        depsi_vx   = -dt * np.cos(epsi) / den * cur
        depsi_vy   = dt * np.sin(epsi) / den * cur
        depsi_wz   = dt
        depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
        depsi_s    = 0  # Because cur = constant
        depsi_ey   = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

        Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
        Ci[3]    = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x)
        # ===========================
        # ===== Linearize s =========
        # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
        ds_vx   = dt * (np.cos(epsi) / den)
        ds_vy   = -dt * (np.sin(epsi) / den)
        ds_wz   = 0
        ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
        ds_s    = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
        ds_ey   = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * (-cur)

        Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
        Ci[4]    = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x)

        # ===========================
        # ===== Linearize ey ========
        # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
        dey_vx   = dt * np.sin(epsi)
        dey_vy   = dt * np.cos(epsi)
        dey_wz   = 0
        dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
        dey_s    = 0
        dey_ey   = 1

        Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
        Ci[5]    = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x)

        endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

        return Ai, Bi, Ci

    def compute_Q_M(self, inputFeatures, usedIt):
        Counter = 0
        X0   = np.empty((0,len(self.stateFeatures)+len(inputFeatures)))
        Ktot = np.empty((0))

        for it in usedIt:
            X0 = np.append( X0, np.hstack((self.xStored[it][np.ix_(self.indexSelected[Counter], self.stateFeatures)],self.uStored[it][np.ix_(self.indexSelected[Counter], inputFeatures)])), axis=0 )
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        M = np.hstack( (X0, np.ones((X0.shape[0], 1))) )
        Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
        Q = matrix(Q0 + self.lamb * np.eye(Q0.shape[0]))

        return Q, M

    def compute_b(self, yIndex, usedIt, M):
        Counter = 0
        y = np.empty((0))
        Ktot = np.empty((0))

        for it in usedIt:
            y       = np.append(y, np.squeeze(self.xStored[it][self.indexSelected[Counter] + 1, yIndex]))
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
        return b

    def LMPC_LocLinReg(self, Q, b, inputFeatures):
        # # Solve QP
        # res_cons = qp(Q, b) # This is ordered as [A B C]
        # # Unpack results
        # result = np.squeeze(np.array(res_cons['x']))
        result = (-np.linalg.pinv(Q, hermitian=True) @ b).squeeze()
        A = result[0:len(self.stateFeatures)]
        B = result[len(self.stateFeatures):(len(self.stateFeatures)+len(inputFeatures))]
        C = result[-1]
        return A, B, C

    def computeIndices(self, x, it):
        oneVec = np.ones( (self.xStored[it].shape[0]-1, 1) )
        xVec = (np.dot( np.array([x]).T, oneVec.T )).T
        DataMatrix = np.hstack((self.xStored[it][0:-1, self.stateFeatures], self.uStored[it][0:-1, :]))

        diff  = np.dot(( DataMatrix - xVec ), self.scaling)
        norm = la.norm(diff, 1, axis=1)
        indexTot =  np.squeeze(np.where(norm < self.h))
        if (indexTot.shape[0] >= self.MaxNumPoint):
            index = np.argsort(norm)[0:self.MaxNumPoint]
        else:
            index = indexTot

        K  = ( 1 - ( norm[index] / self.h )**2 ) * 3/4
        # if norm.shape[0]<500:
        #     self.print_method("norm: ", norm, norm.shape)

        return index, K

def Regression(trajectory, lamb):
    """Estimates linear system dynamics
    x, u: date used in the regression
    lamb: regularization coefficient
    """
    x,u,_ = unpack_trajectory(trajectory)

    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    Y = x[2:x.shape[0], :]
    X = np.hstack((x[1:(x.shape[0] - 1), :], u[1:(x.shape[0] - 1), :]))

    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)

    A = W.T[:, 0:6]
    B = W.T[:, 6:8]

    ErrorMatrix = np.dot(X, W) - Y
    ErrorMax = np.max(ErrorMatrix, axis=0)
    ErrorMin = np.min(ErrorMatrix, axis=0)
    Error = np.vstack((ErrorMax, ErrorMin))

    return A, B, Error

def unpack_state(state : VehicleState):
    x = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
    u = np.array([state.u.u_steer, state.u.u_a])
    x_glob = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.e.psi, state.x.x, state.x.y])

    return x, u, x_glob

def pack_state(x, u, x_glob):
    state = VehicleState()
    state.v.v_long = x[0]
    state.v.v_tran = x[1]
    state.w.w_psi = x[2]
    state.p.e_psi = x[3]
    state.p.s = x[4]
    state.p.x_tran = x[5]
    state.u.u_steer = u[0]
    state.u.u_a = u[1]
    state.e.psi = x_glob[3]
    state.x.x = x_glob[4]
    state.x.y = x_glob[5]

    return state

def unpack_trajectory(trajectory : List[VehicleState]):
    x = np.array([[t.v.v_long, t.v.v_tran, t.w.w_psi, t.p.e_psi, t.p.s, t.p.x_tran] for t in trajectory])
    u = np.array([[t.u.u_steer, t.u.u_a] for t in trajectory])
    x_glob = np.array([[t.v.v_long, t.v.v_tran, t.w.w_psi, t.e.psi, t.x.x, t.x.y] for t in trajectory] )

    return x, u, x_glob

def unpack_safe_set(ss, map):
    s_array = ss[4,:]
    x_tran_array = ss[5,:]
    itr = min(len(s_array), len(x_tran_array))
    ss_typed = []
    for i in range(itr):
        #coords = VehicleCoords(s = s_array[i], x_tran = x_tran_array[i], e_psi = 0)
        #map.local_to_global_typed(coords)

        glob = map.local_to_global((s_array[i], x_tran_array[i], 0))
        coords = VehicleState(x=Position(x=glob[0] ,y=glob[1]), e=OrientationEuler(psi=0))

        ss_typed.append(coords)
    return ss_typed


def unpack_prediction(prediction,map):
    return unpack_safe_set(prediction,map)


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle
