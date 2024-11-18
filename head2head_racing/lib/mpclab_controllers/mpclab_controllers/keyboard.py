#!/usr/bin python3

import numpy as np
import time
import pynput

from mpclab_controllers.abstract_controller import AbstractController

from mpclab_common.pytypes import VehicleState

class KeyboardController(AbstractController):

    def __init__(self):
        self.controller = pynput.keyboard.Listener(on_press = self.on_press,
                                                   on_release = self.on_release)
        self.throttleUp = 'w'
        self.throttleDown = "s"
        self.steerLeft = "a"
        self.steerRight = "d"
        self.throttle = 0.0
        self.steer = 0.0
        self.controller.start()

    def initialize(self):
        return

    def solve(selfs, **args):
        return

    def on_press(self, key):
        if key.char == self.throttleUp:
            self.throttle = .5
        elif key.char == self.steerLeft:
            self.steer = 0.5
        elif key.char == self.steerRight:
            self.steer = -0.5

    def on_release(self, key):
        if key.char == self.steerLeft or key == self.steerRight:
            self.steer = 0.0
        elif key.char == self.throttleUp:
            self.throttle = 0.0

    def step(self, vehicle_state: VehicleState, env_state = None):

        vehicle_state.u.u_a     = self.throttle
        vehicle_state.u.u_steer = self.steer
        return
