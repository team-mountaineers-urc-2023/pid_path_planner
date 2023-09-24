#!/usr/bin/env python3

from collections import namedtuple
from copy import deepcopy
from math import atan2, pi, sqrt, asin, radians
from statistics import mean
from threading import Lock

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose, Point
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

from planner_interfacing.srv import SetPrecision, SetPrecisionRequest, SetPrecisionResponse

from pid_controller import PID_Controller

### helpers ##################################################################

StampedPose = namedtuple('StampedPose', 'pose time')

EulerAngle = namedtuple('EulerAngle', 'roll pitch yaw')

def clamp(value: float, lower: float, upper: float) -> float:
	return min(upper, max(value, lower))

def euler_from_quaternion(x: float, y: float, z: float, w: float) -> EulerAngle:
	"""Return euler angles (roll, pitch, yaw) of given quaternion angle."""

	t0 = 2 * (w * x + y * z)
	t1 = 1 - 2 * (x * x + y * y)
	roll = atan2(t0, t1)

	t2 = 2 * (w * y - z * x)
	t2 = clamp(t2, -1, 1)
	pitch = asin(t2)

	t3 = 2 * (w * z + x * y)
	t4 = 1 - 2 * (y * y + z * z)
	yaw = atan2(t3, t4)

	return EulerAngle(roll, pitch, yaw)

def npi_to_pi_angle(angle):
	if angle > pi:
		return angle - 2*pi
	elif angle < -pi:
		return angle + 2*pi
	return angle

def pose_to_waypoint_error(pose: Pose, waypoint: Point):
	# find the error in cartesian coordinates
	x_error = waypoint.x - pose.position.x
	y_error = waypoint.y - pose.position.y
	dist_error = sqrt(x_error**2 + y_error**2)

	# find heading
	ori = pose.orientation
	euler = euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)
	heading = euler.yaw
	
	# calculate the heading that would point at the waypoint
	heading_to_waypoint = atan2(y_error, x_error)

	# find the heading error (choose smaller direction of rotation)
	heading_error = heading_to_waypoint - heading
	heading_error = npi_to_pi_angle(heading_error)

	return dist_error, heading_error

def dist2d(pt1: Point, pt2: Point):
	x_error = pt1.x - pt2.x
	y_error = pt1.y - pt2.y
	return sqrt(x_error**2 + y_error**2)

### main #####################################################################

def main():
	PID_Planner().loop()

class PID_Planner:
	def __init__(self) -> None:

		rospy.init_node('pid_planner')

		### prepare PID controllers ##########################################

		dist_prop_gain = rospy.get_param("~dist_prop_gain")
		dist_kp = rospy.get_param("~dist_kp")
		dist_kd = rospy.get_param("~dist_kd")
		dist_ki = rospy.get_param("~dist_ki")
		self.dist_pid_controller = PID_Controller(dist_prop_gain, dist_kp, dist_kd, dist_ki)

		theta_prop_gain = rospy.get_param("~theta_prop_gain")
		head_kp = rospy.get_param("~head_kp")
		head_kd = rospy.get_param("~head_kd")
		head_ki = rospy.get_param("~head_ki")
		self.head_pid_controller = PID_Controller(theta_prop_gain, head_kp, head_kd, head_ki)

		self.pid_lock = Lock()
		
		### local variables ##################################################

		self.min_linear_speed = rospy.get_param("~min_linear_speed")
		self.max_linear_speed = rospy.get_param("~max_linear_speed")
		self.min_angular_speed = rospy.get_param("~min_angular_speed")
		self.max_angular_speed = rospy.get_param("~max_angular_speed")

		self.angle_threshold = radians(rospy.get_param("~angle_threshold"))
		default_precision = rospy.get_param("~default_precision")
		self.freq = rospy.get_param("~update_frequency")
		self.not_moving_threshold = rospy.get_param("~not_moving_threshold")
		self.not_moving_delay = rospy.Duration(rospy.get_param("~not_moving_delay"))
		self.stuck_delay = rospy.Duration(rospy.get_param("~stuck_delay"))

		self.enabled_lock = Lock()
		self.enabled = False  # whether or not to publish cmd_vel

		self.precision_lock = Lock()
		self.precision = default_precision

		self.start_move_time_lock = Lock()
		self.start_move_time = None

		self.waypoint_lock = Lock()
		self.current_waypoint = None

		self.pose_lock = Lock()
		self.current_pose = None

		self.recent_positions_window_lock = Lock()
		self.recent_positions_window = []

		self.not_moving_lock = Lock()
		self.not_moving = True

		self.stuck_lock = Lock()
		self.stuck = False

		### connect to ROS ###################################################

		current_waypoint_topic = rospy.get_param("~current_waypoint_topic")
		local_position_topic = rospy.get_param("~local_position_topic")
		cmd_vel_topic = rospy.get_param("~cmd_vel_topic")
		current_status_topic = rospy.get_param("~current_status_topic")
		enabled_service_name = rospy.get_param("~enabled_service")
		precision_service_name = rospy.get_param("~precision_service")

		self.current_waypoint_sub = rospy.Subscriber(current_waypoint_topic, Point, self.current_waypoint_sub_callback)
		self.pose_sub = rospy.Subscriber(local_position_topic, Pose, self.pose_sub_callback)
		self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
		self.current_status_pub = rospy.Publisher(current_status_topic, String, queue_size=1)
		self.enabled_service = rospy.Service(enabled_service_name, SetBool, self.enabled_callback)
		self.precision_service = rospy.Service(precision_service_name, SetPrecision, self.precision_callback)

		### end init #########################################################

	### local functions ######################################################

	def publish_status(self):
		# +------------+----------+-------------+
		# |            | waypoint | no waypoint |
		# +------------+-----------+------------+
		# | ability    | active   | idle        |
		# | no ability | blocked  | inactive    |
		# +------------+----------+-------------+

		with self.enabled_lock:
			enabled = self.enabled
		with self.waypoint_lock:
			has_waypoint = self.current_waypoint is not None
		with self.pose_lock:
			has_location = self.current_pose is not None
		with self.stuck_lock:
			stuck = self.stuck

		has_ability = enabled and has_location

		if has_ability and has_waypoint:
			if stuck:
				status = "stuck"
			else:
				status = "active"
		elif has_ability and not has_waypoint:
			status = "idle"
		elif not has_ability and has_waypoint:
			status = "blocked"
		elif not has_ability and not has_waypoint:
			status = "inactive"
		self.current_status_pub.publish(String(status))

	### callbacks ############################################################

	def current_waypoint_sub_callback(self, waypoint: Point):
		with self.pose_lock:
			current_position = deepcopy(self.current_pose.position)
		with self.waypoint_lock:
			current_waypoint = deepcopy(self.current_waypoint)
		with self.precision_lock:
			precision = deepcopy(self.precision)

		# check if already at waypoint
		at_waypoint = dist2d(current_position, waypoint) < precision

		# check if waypoint is significantly different than current target
		new_target = not current_waypoint or dist2d(waypoint, current_waypoint) > precision

		if not at_waypoint:
			# update target waypoint
			with self.waypoint_lock:
				self.current_waypoint = waypoint

		if not at_waypoint and new_target:
			# reset pid errors
			with self.pid_lock:
				self.dist_pid_controller.reset()
				self.head_pid_controller.reset()
			with self.start_move_time_lock:
				self.start_move_time = rospy.Time.now()

		self.publish_status()

	def enabled_callback(self, bool: SetBoolRequest) -> SetBoolResponse:
		enabled = bool.data
		with self.enabled_lock:
			being_disabled = self.enabled and not enabled
			being_enabled = not self.enabled and enabled
			self.enabled = enabled
		self.publish_status()

		if being_disabled:
			self.cmd_vel_pub.publish(Twist())

		if being_enabled:
			with self.start_move_time_lock:
				self.start_move_time = rospy.Time.now()

		response = SetBoolResponse()
		response.success = True
		response.message = "Updated pid path planner's enable status"
		return response

	def precision_callback(self, precision_req: SetPrecisionRequest) -> SetPrecisionResponse:
		precision = precision_req.precision
		with self.precision_lock:
			self.precision = precision

		response = SetPrecisionResponse()
		response.success = True
		response.message = f"Updated pid path planner's precision to {precision}"
		return response

	def pose_sub_callback(self, pose_msg: Pose):
		with self.pose_lock:
			self.current_pose = pose_msg

		with self.recent_positions_window_lock:
			# add position to recent position window
			self.recent_positions_window.append(StampedPose(pose_msg, rospy.Time.now()))

			# remove positions too old to be in recent position window
			updated_window = []
			for stamped_pose in self.recent_positions_window:
				if stamped_pose.time + self.not_moving_delay > rospy.Time.now():
					updated_window.append(stamped_pose)

			self.recent_positions_window = updated_window

		# check if we are not moving by comparing current position to average position of position window
		with self.not_moving_lock:
			positions_in_window = [stamped_pose.pose.position for stamped_pose in updated_window]
			x_mean = mean([position.x for position in positions_in_window] or [0])
			y_mean = mean([position.y for position in positions_in_window] or [0])
			z_mean = mean([position.z for position in positions_in_window] or [0])
			average_position = Point(x_mean, y_mean, z_mean)
			self.not_moving = dist2d(average_position, pose_msg.position) * 2 < self.not_moving_threshold

	### loop #################################################################

	def loop(self):
		rate = rospy.Rate(self.freq)

		while not rospy.is_shutdown():

			# ensure pid errors are not reset until after current error calculations are applied
			with self.pid_lock:

				with self.enabled_lock:
					enabled = self.enabled
				with self.precision_lock:
					precision = self.precision
				with self.waypoint_lock:
					current_waypoint = deepcopy(self.current_waypoint)
				with self.pose_lock:
					current_pose = deepcopy(self.current_pose)
				with self.not_moving_lock:
					not_moving = self.not_moving
				with self.start_move_time_lock:
					start_move_time = self.start_move_time
				if enabled and current_waypoint and current_pose:

					# calculate error between current pose and waypoint
					dist_error, head_error = pose_to_waypoint_error(current_pose, current_waypoint)

					# if waypoint has been reached, remove it and set status to idle
					if dist_error < precision:
						with self.waypoint_lock:
							self.current_waypoint = None
						self.cmd_vel_pub.publish(Twist())
						self.publish_status()
						continue

					# update pid errors and receive velocity signals
					linear = self.dist_pid_controller.update(dist_error)
					angular = self.head_pid_controller.update(head_error)

					# dont move forward if we still need to turn a lot
					if abs(head_error) > self.angle_threshold:
						linear = 0.0

					# check if we are stuck
					should_be_moving_by_now = rospy.Time.now() > start_move_time + self.stuck_delay
					stuck = not_moving and should_be_moving_by_now
					with self.stuck_lock:
						self.stuck = stuck

					# move at full speed forward if stuck
					if stuck:
						linear, angular = self.max_linear_speed, 0.0

					# scale cmd_vel and publish
					cmd_vel = Twist()
					cmd_vel.linear.x = clamp(linear, self.min_linear_speed, self.max_linear_speed)
					cmd_vel.angular.z = clamp(angular, self.min_angular_speed, self.max_angular_speed)
					self.cmd_vel_pub.publish(cmd_vel)
					self.publish_status()

			# run the loop at the specified rate
			rate.sleep()

if __name__ == '__main__':
	main()
