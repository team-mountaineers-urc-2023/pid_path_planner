#!/usr/bin/env python3

from time import time

class PID_Controller():
	def __init__(self, prop_gain, kp, kd, ki):
		self.prop_gain = prop_gain
		self.kp = kp
		self.kd = kd
		self.ki = ki

		self.last_time = time()
		self.previous_error = 0
		self.previous_error_sum = 0

		self.val = 0

	def update(self, current_error: float) -> float:
		# proportional contribution
		result_p = self.kp * current_error

		# derivative contribution
		curr_time = time()
		result_d = self.kd * ((current_error - self.previous_error) / (curr_time - self.last_time))

		# integral contribution
		self.previous_error_sum += self.previous_error
		result_i = self.ki * self.previous_error_sum

		# update error history
		self.previous_error = current_error
		self.last_time = curr_time

		result = result_p + result_d + result_i
		self.val = result * self.prop_gain
		return self.val

	def reset(self):
		self.last_time = time()
		self.previous_error = 0
		self.previous_error_sum = 0
