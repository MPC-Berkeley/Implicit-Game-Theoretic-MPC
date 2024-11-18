#!/usr/bin python3

import numpy as np
import time
from xbox360controller import Xbox360Controller

from mpclab_common.pytypes import VehicleState

from mpclab_controllers.abstract_controller import AbstractController
from mpclab_controllers.utils.controllerTypes import PIDParams, JoystickParams
from mpclab_controllers.PID import PIDLaneFollower

class XboxJoystick(AbstractController):
    '''
    Class for PID throttle control and open loop steering control of a vehicle. Can be used to do identification of steering map
    Incorporates separate PID controllers for maintaining a constant speed

    target speed: v_ref
    '''
    def __init__(self, params=JoystickParams()):
        self.controller = Xbox360Controller(0, axis_threshold=0.0)

        self.dt = params.dt

        self.u_steer_max = params.u_steer_max
        self.u_steer_min = params.u_steer_min
        self.u_steer_neutral = params.u_steer_neutral
        self.u_steer_rate_max = params.u_steer_rate_max
        self.u_steer_rate_min = params.u_steer_rate_min

        self.u_a_max = params.u_a_max
        self.u_a_min = params.u_a_min
        self.u_a_neutral = params.u_a_neutral
        self.u_a_rate_max = params.u_a_rate_max
        self.u_a_rate_min = params.u_a_rate_min

        self.u_a_prev = 0
        self.u_steer_prev = 0

        self.throttle_pid = params.throttle_pid
        self.steering_pid = params.steering_pid

        self.throttle_pid_params = params.throttle_pid_params
        self.steering_pid_params = params.steering_pid_params

        self.pid = PIDLaneFollower(self.dt, self.throttle_pid_params, self.steering_pid_params)
        self.pid.steer_pid.set_u_ref(self.u_steer_neutral)
        self.pid.speed_pid.set_u_ref(self.u_a_neutral)

        self.e_brake = False

        self.controller.button_a.when_pressed = self.e_brake_engage
        self.controller.button_a.when_released = self.e_brake_disengage

        return

    def initialize(self, **args):
        return

    def solve(self, **args):
        return

    def e_brake_engage(self, button):
        self.e_brake = True

    def e_brake_disengage(self, button):
        self.e_brake = False

    def step(self, vehicle_state: VehicleState, env_state = None):
        self.pid.step(vehicle_state)

        if not self.throttle_pid:
            throttle_input = self.controller.trigger_r.value
            brake_input = self.controller.trigger_l.value
            if brake_input > 0:
                u_a = self.u_a_neutral + brake_input * (self.u_a_min - self.u_a_neutral)
            else:
                u_a = self.u_a_neutral + throttle_input * (self.u_a_max - self.u_a_neutral)

            if self.u_a_rate_max is not None:
                u_a = self.u_a_prev + min(self.u_a_rate_max*self.dt, u_a - self.u_a_prev)
            if self.u_a_rate_min is not None:
                u_a = self.u_a_prev + max(self.u_a_rate_min*self.dt, u_a - self.u_a_prev)

        if not self.steering_pid:
            steering_input = -self.controller.axis_l.x
            if steering_input >= 0:
                u_steer = self.u_steer_neutral + steering_input * (self.u_steer_max - self.u_steer_neutral)
            else:
                u_steer =  self.u_steer_neutral - steering_input * (self.u_steer_min - self.u_steer_neutral)

            if self.u_steer_rate_max is not None:
                u_steer = self.u_steer_prev + min(self.u_steer_rate_max*self.dt, u_steer - self.u_steer_prev)
            if self.u_steer_rate_min is not None:
                u_steer = self.u_steer_prev + max(self.u_steer_rate_min*self.dt, u_steer - self.u_steer_prev)

        vehicle_state.u.u_a = u_a
        vehicle_state.u.u_steer = u_steer

        self.u_a_prev = u_a
        self.u_steer_prev = u_steer

        return

    def close(self):
        self.controller.close()
