# -*- coding: utf-8 -*-
import random
import time
from datetime import datetime
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import open3d as o3d
import weakref
import math
import cv2
import gymnasium as gym
from gym import spaces

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

import carla

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from PythonAPI.agents.navigation.global_route_planner import GlobalRoutePlanner
from PythonAPI.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from PythonAPI.agents.navigation.basic_agent import BasicAgent
from PythonAPI.agents.toolkit.planner.random_planner import RandomPlanner

#import sheeprl's dreamerV3 components
from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import prepare_obs
from sheeprl.utils.utils import symlog_inv

#show training process
showPreview = True

# ~~~~~~~~~~ SCENARIO CONFIG ~~~~~~~~~~ #
#the same env file is used both to train DreamerV3 (sheeprl training run) and to evaluate the frozen agent
#(utils.test), the globals below select what the env does in either case

#route scenario, which town is loaded and how the spawn point / destination of every episode are chosen (paper S1 / S2 / S3)
#  "static"       -> STATIC_TOWN is loaded, every episode starts at STATIC_SPAWN_LOCATION and drives to STATIC_DESTINATION
#  "random"       -> RANDOM_TOWN is loaded, every episode starts at a random spawn point with a ROUTE_LENGTH_M route ahead
#  "random_towns" -> a random town of TOWN_LIST is loaded, random spawn point and route, and the town is
#                    swapped for another random one of TOWN_LIST every TOWN_CHANGE_EVERY episodes
ROUTE_SCENARIO = "random_towns"

#obstacle regime, what is spawned on the route at the start of every episode (paper R1 / R2)
#  "none"   -> nothing is spawned. USE THIS WHEN TRAINING THE AGENT: the world model must only ever see
#              obstacle free scenes, the anomaly detectors rely on the agent being anomaly agnostic
#  "single" -> one static car (SINGLE_OBSTACLE_BLUEPRINT) on a random route point, evaluation regime R1
#  "random" -> a random vehicle / walker / static prop on a random route point, evaluation regime R2
OBSTACLE_REGIME = "random"

#"static" scenario: town and fixed coordinates (x, y, z in CARLA world frame, meters)
STATIC_TOWN = 'Town01'                                          #town loaded for the static scenario
STATIC_SPAWN_LOCATION = (87.687683, 145.671295, 0.300000)       #where the ego vehicle is spawned
STATIC_SPAWN_YAW = 90.000053                                    #heading of the ego vehicle at spawn, degrees
STATIC_INITIAL_LOCATION = (88.392143, 145.671188, 0.000000)     #start point handed to the route planner (on the road, next to the spawn)
STATIC_DESTINATION = (162.245148, 330.596283, 0.000000)         #end point of the fixed route

#"random" and "random_towns" scenarios
RANDOM_TOWN = 'Town01'                                          #town loaded for the "random" scenario (single town, random routes)
TOWN_LIST = ['Town01','Town02','Town03','Town04','Town05']      #towns the "random_towns" scenario cycles through
TOWN_CHANGE_EVERY = 20                                          #"random_towns": a new town is loaded every this many episodes
ROUTE_LENGTH_M = 250                                            #random routes: distance in meters from the spawn point to the destination waypoint
MIN_ROUTE_WAYPOINTS = 100                                       #random routes: a new spawn point is drawn until the planned route has at least this many waypoints

#obstacle spawn (ignored when OBSTACLE_REGIME is "none")
SINGLE_OBSTACLE_BLUEPRINT = 'vehicle.tesla.model3'              #CARLA blueprint of the obstacle in the "single" regime (a car, class 14)
RANDOM_OBSTACLE_FILTERS = ['vehicle.*', 'static.*.*']           #CARLA blueprint filters of the "random" regime pool. walkers are left out: a pedestrian hit by the
                                                                #ego vehicle is pushed along without a collision event in CARLA, so such episodes cannot end in a collision
OBSTACLE_MIN_WAYPOINT = 10                                      #the obstacle is placed on a route waypoint at least this many waypoints after the start and before the end,
                                                                #so every episode begins obstacle free and becomes anomalous once the obstacle enters the camera
OBSTACLE_STRAIGHT_LOOKBACK = 15                                 #the obstacle is placed on a straight approach: the route heading over this many waypoints before the
OBSTACLE_MAX_APPROACH_TURN_DEG = 15.0                           #obstacle may change by at most this many degrees, so the obstacle is in view before the car reaches it.
OBSTACLE_PLACEMENT_TRIES = 50                                   #random route indices tried for a straight approach before the constraint is dropped for the episode

#safe stop termination, EVALUATION ONLY. when the safety supervisor (utils.AnomalyObservationSystem) has forced the
#brake and the car has come to a stop, the episode is truncated instead of idling until the step horizon.
#it never triggers during training because nothing sets the brake there, so the agent's training is unaffected
SAFE_STOP_TERMINATION = True                                    #False -> a stopped car idles until the step horizon like before
SAFE_STOP_SPEED = 0.1                                           #m/s, the car counts as stopped below this speed
SAFE_STOP_STEPS = 10                                            #the car must be stopped with the brake engaged for this many consecutive steps (10 = 1 s at 10 Hz) before the episode ends

#gymnasium environment for CARLA tailored to DreamerV3 (paper Sec. IV-A).
#observation: the semantically segmented roof camera and a 12 dim state vector in the vehicle frame,
#action: 15 discrete acceleration x steering pairs, reward: waypoint following, target speed, collision, lane keeping.
#CarEnv holds everything that is independent of the simulator connection, ContinuousCarEnv connects to CARLA
class CarEnv(gym.Env, ABC):
    #declares the observation space and the state the evaluation loop reads / writes (imagination, safety command)
    @abstractmethod
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (128, 128, 3),
        n_steps: int = 500,
        dict_obs_space: bool = True,
    ):
        # set observation space to single roof camera
        # Dreamer requires a flatten dictionary as an observation space instead of a simple box
        self.observation_space = gym.spaces.Dict(
          {
             "semantic_camera": gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
             # state vector layout (12 dims):
             #   [0:10] = 5 lookahead waypoints in vehicle frame, each (forward_dist, lateral_dist)
             #   [10]   = speed_norm
             #   [11]   = angle_deg to immediate next waypoint
             "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
           }
         )

        self.reward_range = (-np.inf, np.inf)
        self._current_step = 0
        self._n_steps = n_steps
        self._render_mode = "rgb_array"
        self._imagination = None
        self._observation = None
        self._imagination_variance = None
        self.current_episode = 0
        self.anomaly_detection = 0.0

        #safety supervisor command, neutral unless set_safety_command() is called
        self.safety_brake = 0.0
        self.safety_throttle_scale = 1.0
        self.stopped_steps = 0

    #cleanup function, stops and destroys every sensor and actor of the episode. called on every episode end
    def close(self):
      #print("~~~~~~~~~~~~~~~~")
      #print("Close Env")
      try:
        if len(self.actor_list) !=0 or len(self.sensor_list) !=0:
          for sensor in self.sensor_list:
            if sensor.is_listening:
              sensor.stop()
              time.sleep(0.1)
            sensor.destroy()

          for actor in self.actor_list:
            actor.destroy()

          #remove the actors from the list and clear the sensor data dictionary
          self.actor_list.clear()
          self.sensor_list.clear()

          cv2.destroyAllWindows() # close opencv windows

      except:
        for sensor in self.sensor_list:
          sensor.destroy()

        for actor in self.actor_list:
          actor.destroy()

        self.sensor_list.clear()
        self.actor_list.clear()
        
    #decodes the discrete action index into throttle / brake / steer (paper eq. u : A -> [0,1]^2 x [-1,1])
    #and applies the safety supervisor override on top of the agent's command
    def get_action(self, action):
        acc = self.discrete_acc[action // self.n_steer]
        steer = self.discrete_steer[action % self.n_steer]
        
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        #safety supervisor override (anomaly observation system)
        #safety_brake 0..1 forces the brake, 1 = brake to a stop (obstacle detected)
        #safety_throttle_scale 0..1 scales the throttle (only used by the standalone kl mode with S2_ACTION "cautious")
        if self.safety_brake > 0 or self.safety_throttle_scale < 1:
            throttle = throttle * self.safety_throttle_scale * (1 - self.safety_brake)
            brake = max(brake, self.safety_brake)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    #reward function (paper eq. 6 - 7): waypoint, speed, collision, lane invasion, destination and time terms.
    #also decides the terminal / truncation conditions and builds the state observation of the current step
    def getReward(self):
      reward = 0
      terminated = False
      truncated = False

      # ----- Vehicle pose (always computed, needed for state obs even when route is empty) -----
      vehicle_location = self.vehicle.get_location()
      vehicle_transform = self.vehicle.get_transform()
      v_loc = vehicle_transform.location
      v_rot = vehicle_transform.rotation
      forward = v_rot.get_forward_vector()

      # 2D forward (unit) and right (unit) vectors in world frame, used for vehicle-frame projection
      fwd_2d = np.array([forward.x, forward.y], dtype=np.float32)
      fwd_norm = np.linalg.norm(fwd_2d) + 1e-6
      fwd_2d = fwd_2d / fwd_norm
      # right-hand perpendicular in 2D (rotate forward by -90 deg around z)
      right_2d = np.array([fwd_2d[1], -fwd_2d[0]], dtype=np.float32)

      velocity_vector = self.vehicle.get_velocity()
      velocity_x = round(velocity_vector.x, 6)
      velocity_y = round(velocity_vector.y, 6)
      velocity = np.array([velocity_x, velocity_y])
      speed_norm = np.linalg.norm(velocity)

      distance_from_next_point = self.next_point.distance(vehicle_location)

      #distance to the injected obstacle (nan when there is no obstacle)
      obstacle_distance = float("nan")
      if self.spawn_npc_location is not None:
        obstacle_distance = vehicle_location.distance(self.spawn_npc_location)

      distance_between_points = self.prev_point.distance(self.next_point)
      distance_traveled_to_next_point = distance_from_next_point - distance_between_points
      self.distance_travelled = self.initial_location.distance(vehicle_location)

      # default values (used when route is empty / on the terminal step)
      navigation_reward = 0.0
      distance_reward = 0.0
      speed_reward = 0.0
      angle_reward = 0.0
      angle_deg = 0.0
      speed_parallel = 0.0
      speed_perpendicular = 0.0
      perp_direction_norm = 0.0

      if len(self.route) > 0:
        # ----- Angle from vehicle heading to immediate next waypoint -----
        dx = self.next_point.x - v_loc.x
        dy = self.next_point.y - v_loc.y

        dot = (forward.x * dx) + (forward.y * dy)
        cross_z = (forward.x * dy) - (forward.y * dx)

        angle_rad = math.atan2(cross_z, dot)
        angle_deg = math.degrees(angle_rad)

        # ----- Waypoint-line direction -----
        # FIX: original code used `next_point.z * pi/180` as yaw, but `next_point` is a
        # carla.Location whose .z is height (0.3), not yaw. This made waypoint_direction
        # ~constant in world-X across every step, breaking the speed reward in random routes.
        # Replacement: direction along the segment from prev_point -> next_point.
        seg_dx = self.next_point.x - self.prev_point.x
        seg_dy = self.next_point.y - self.prev_point.y
        seg_norm = math.hypot(seg_dx, seg_dy) + 1e-6
        waypoint_direction = np.array([seg_dx / seg_norm, seg_dy / seg_norm], dtype=np.float32)

        # ----- Perpendicular deviation (lateral offset) from the route line -----
        # FIX: original code mixed a scalar (distance_from_next_point) with a vector,
        # producing meaningless perp_direction_norm. Use proper 2D displacement vector.
        disp = np.array([self.next_point.x - v_loc.x,
                         self.next_point.y - v_loc.y], dtype=np.float32)
        along = float(np.dot(disp, waypoint_direction))
        perp_vec = disp - along * waypoint_direction
        perp_direction_norm = float(np.linalg.norm(perp_vec))

        if perp_direction_norm > 0.05:
          perp_unit = perp_vec / perp_direction_norm
        else:
          perp_unit = np.array([0.0, 0.0], dtype=np.float32)

        # ----- Speed reward -----
        speed_parallel = float(np.dot(velocity, waypoint_direction))      # towards waypoint
        speed_perpendicular = float(np.abs(np.dot(velocity, perp_unit)))  # drifting from lane
        speed_reward = (self.desired_speed
                        - np.abs(speed_parallel - self.desired_speed)
                        - 2 * min(speed_perpendicular, 0.5)) * self.speed_weight

      #punish for collision
      collision_reward = 0.0
      if len(self.collision_hist) != 0:
        collision_reward = -self.collision_weight #* np.abs(speed_norm)
        terminated = True
        self.close()

      #punish for lane invation
      lane_invasion_reward = 0.0
      if len(self.lane_invade_hist) != 0:
        if perp_direction_norm > 0.5:
           lane_invasion_reward = -self.lane_invasion_weight * (perp_direction_norm - 0.5)
        else:
           lane_invasion_reward = -self.lane_invasion_weight
        terminated = True
        self.lane_invade_hist.clear()
        self.close()

      # ----- Punish for missing a waypoint -----
      # FIX: threshold loosened from 8m to 15m. With random spawn points (often at
      # intersections) the agent needs more margin during exploration; 8m was killing
      # episodes before any learning could happen.
      waypoint_reward = 0.0
      if distance_from_next_point > 15:
        waypoint_reward = -self.waypoint_weight*2
        terminated = True
        self.close()

      #reward for reaching a waypoint
      #2.2 distance threshold
      if distance_from_next_point < 1.8 and speed_norm > 0:
        self.waypoints_reached += 1
        waypoint_reward = self.waypoint_weight
        self.prev_point = carla.Location(x=self.route[0][0], y=self.route[0][1], z=0.0)
        self.route.pop(0)

      #reward for reaching the destination
      destination_reward = 0.0
      distance_from_destination = vehicle_location.distance(self.destination)

      if len(self.route) == 0:
        destination_reward = self.destination_reached_weight
        terminated = True
        self.close()

      #time penalty
      time_reward = -self.time_weight

      reward = waypoint_reward + speed_reward + collision_reward + lane_invasion_reward + destination_reward + time_reward
      
      #check for episode duration
      if self._current_step == self.total_timesteps :
        truncated = True
        self.close()

      #evaluation only: the safety supervisor has stopped the car, end the episode
      safe_stop = False
      if SAFE_STOP_TERMINATION and self.safety_brake > 0 and speed_norm < SAFE_STOP_SPEED:
        self.stopped_steps += 1
        if self.stopped_steps >= SAFE_STOP_STEPS:
          safe_stop = True
          truncated = True
          self.close()
      else:
        self.stopped_steps = 0
      
      self.terminal_conditions = {
        "collision": np.array(len(self.collision_hist) > 0, dtype=bool),
        "time_exceeded": np.array(self._current_step == self.total_timesteps, dtype=bool),
        "lane_invasion": np.array(len(self.lane_invade_hist) > 0, dtype=bool),
        "destination_reached": np.array(distance_from_destination < 2.2, dtype=bool),
        "safe_stop": np.array(safe_stop, dtype=bool),
      }

      #check agent info at some point
      self.reward_info = {
        "ego_x": vehicle_location.x,
        "ego_y": vehicle_location.y,
        "speed_parallel": speed_parallel,
        "speed_perpendicular": speed_perpendicular,
        "speed_norm": speed_norm,
        "angle": angle_deg,
        "waypoint_distance": distance_from_next_point,
        "waypoint_reward": waypoint_reward,
        "speed_reward": speed_reward,
        "angle_reward": angle_reward,
        "collision_reward": collision_reward,
        "lane_invasion_reward": lane_invasion_reward,
        "destination_reward": destination_reward,
        "time_reward": time_reward,
        "obstacle_distance": obstacle_distance,
        "obstacle_type": self.npc_blueprint_id,
        "safety_brake": self.safety_brake,
        "safety_throttle_scale": self.safety_throttle_scale,
      }
      
      # ----- State observation: vehicle-frame lookahead -----
      # FIX: original passed absolute world coordinates of vehicle and the immediate next
      # waypoint. With a static spawn this was effectively a fixed reference frame, but
      # with random spawn points the same coordinate values mean different things across
      # episodes, and the agent had no information about waypoints beyond the immediate
      # next one (impossible to plan turns at intersections).
      # Replacement: 5 lookahead waypoints expressed in the vehicle's own frame
      # (forward_dist, lateral_dist), padded with the last available waypoint if the
      # remaining route is shorter than the lookahead window.
      LOOKAHEAD = 5
      lookahead_features = []
      for i in range(LOOKAHEAD):
          if i < len(self.route):
              wp_x, wp_y = self.route[i]
          elif len(self.route) > 0:
              wp_x, wp_y = self.route[-1]  # pad with last remaining waypoint
          else:
              wp_x, wp_y = self.next_point.x, self.next_point.y  # final terminal step
          rel_dx = wp_x - v_loc.x
          rel_dy = wp_y - v_loc.y
          fwd_dist = rel_dx * fwd_2d[0]   + rel_dy * fwd_2d[1]
          lat_dist = rel_dx * right_2d[0] + rel_dy * right_2d[1]
          lookahead_features.extend([fwd_dist, lat_dist])

      self.state_observation = np.array([
          *lookahead_features,   # 10 dims
          speed_norm,            # 1 dim
          angle_deg,             # 1 dim
      ], dtype=np.float32)

      return terminated, truncated, reward

    #step function: applies the action, ticks the simulator, computes reward and observation, returns the gym tuple
    def step(self, action):
        control = self.get_action(action)
        
        #apply vehicle actions (speed, steering, braking)
        self.vehicle.apply_control(control)
        
        #get spectator cam to follow vehicle
        if self.SHOW_CAM and self.roof_camera_observation is not None:
          #follow vehicle
          self.spectator.set_transform(carla.Transform(self.spawnPoint.location + carla.Location(z=50),carla.Rotation(pitch=-90)))

          #show camera feed
          display_img = self.roof_camera_observation
          
          camera = cv2.resize(display_img, (256, 256), cv2.INTER_NEAREST)
          if self._imagination is not None:
            imag_img = self.imagination_to_rgb(self._imagination)
            imagination = cv2.resize(imag_img, (256, 256), cv2.INTER_NEAREST)
            panel = cv2.hconcat([camera, imagination])
          else:
            panel = cv2.hconcat([camera])

          cv2.imshow("Camera + Imagination", panel)
          cv2.resizeWindow("Camera + Imagination", 512, 512)          

          cv2.waitKey(1)
          
        self.world.tick()
        self._current_step += 1

        self.next_point = carla.Location(x=self.route[0][0], y=self.route[0][1], z=0.3)
        
        #get reward
        terminated, truncated, reward = self.getReward()
                
        self.state = {
          "ego_waypoints": self.route,
          "timesteps": self._current_step
        }
        
        self.obs_info = {
          "terminal": terminated,
          "num_completed": self.waypoints_reached,
          "num_obsolete": self.total_waypoints - self.waypoints_reached,
          "travel_distance": self.distance_travelled
        }
        
        info = {
          **self.state,
          **self.terminal_conditions,
          **self.obs_info,
          **self.reward_info,
          "action": action
        }      
        
        return (
            self.get_obs(),
            reward,
            terminated,
            truncated,
            info
        )

    #returns the observation dictionary (camera + state vector), zeros before the first camera frame arrives
    def get_obs(self) -> Dict[str, np.ndarray]:
        if self.roof_camera_observation is not None:
            return {
                "semantic_camera": self.roof_camera_observation,
                "state": self.state_observation,
            }
        else:
            return {
                "semantic_camera": np.zeros([128, 128, 3], dtype=np.uint8),
                "state": np.zeros([12], dtype=np.float32),
            }

    #heading change (degrees) of the route over the OBSTACLE_STRAIGHT_LOOKBACK waypoints before index i
    def approach_turn_deg(self, i):
      j = max(i - OBSTACLE_STRAIGHT_LOOKBACK, 1)
      h0 = math.atan2(self.route[j][1] - self.route[j - 1][1], self.route[j][0] - self.route[j - 1][0])
      h1 = math.atan2(self.route[i][1] - self.route[i - 1][1], self.route[i][0] - self.route[i - 1][0])
      d = math.degrees(h1 - h0)
      return abs((d + 180) % 360 - 180)

    #random route index for the obstacle: away from the start and the end, and on a straight approach
    def pick_obstacle_index(self):
      lo = max(OBSTACLE_MIN_WAYPOINT, OBSTACLE_STRAIGHT_LOOKBACK + 1)
      hi = len(self.route) - OBSTACLE_MIN_WAYPOINT
      for _ in range(OBSTACLE_PLACEMENT_TRIES):
        i = random.randint(lo, hi)
        if self.approach_turn_deg(i) <= OBSTACLE_MAX_APPROACH_TURN_DEG:
          return i
      print("no straight approach found for the obstacle, placing it on a random route point")
      return random.randint(lo, hi)

    #plans the route of the episode with the global route planner and stores it as a list of (x, y) waypoints
    def get_waypoints(self):
      #print("~~~~~~~~~~~~~~~~")
      #print("Calculate Route")
      if ROUTE_SCENARIO == "static":
        #~~~~~~~~ static route ~~~~~~~~
        self.initial_location = carla.Location(*STATIC_INITIAL_LOCATION)
        self.destination = carla.Location(*STATIC_DESTINATION)
      else:
        #~~~~~~~~ random route ~~~~~~~~
        #get spawn point
        self.initial_location = self.spawnPoint.location
        self.origin_waypoint = self.map.get_waypoint(self.initial_location, project_to_road=True)

        #get destination point
        self.destination_waypoint = self.origin_waypoint.next(ROUTE_LENGTH_M)
        self.destination = self.destination_waypoint[0].transform.location
      time.sleep(0.01)
      
      if not self.destination:
         print("could not find a destination point")
         return

      #~~~~~~~~ make route path ~~~~~~~~
      start_location = [self.initial_location.x, self.initial_location.y, 0.0]
      end_location = [self.destination.x, self.destination.y, 0.0]
      solution = self.agent.set_destination(start_location, end_location)
      
      time.sleep(0.01)
      
      self.route = list()
      
      if not solution:
         return
      
      for item in solution:
         waypoint = item[0]
         self.route.append((waypoint.transform.location.x, waypoint.transform.location.y))
      
      self.prev_point = self.initial_location
      self.next_point = carla.Location(x=self.route[0][0], y=self.route[0][1], z=0.3)

      self.total_waypoints = len(self.route)

      self.distance_to_destination = self.initial_location.distance(self.destination)
      
    #setters called by the evaluation loop (utils.test_mode) every step

    #camera observation as seen by the agent, for the preview window
    def set_observation(self, observation):
      self._observation = observation
      
    #decoder reconstruction of the current scene ("imagination"), shown next to the camera in the preview window
    def set_imagination(self, imagination):
      self._imagination = imagination
      
    #episode counter of the evaluation loop, drives the town change of the random_towns scenario
    def set_current_episode(self, episode):
      self.current_episode = episode
      
    #raw cnn anomaly output of the current frame, stored for logging
    def set_anomaly_detection(self, anomaly_detection):
      self.anomaly_detection = anomaly_detection

    #command from the safety supervisor, applied on the next get_action() call
    def set_safety_command(self, brake=0.0, throttle_scale=1.0):
      self.safety_brake = float(np.clip(brake, 0, 1))
      self.safety_throttle_scale = float(np.clip(throttle_scale, 0, 1))
      
    #converts a decoder output (1, 1, 3, H, W) in [-0.5, 0.5] back to an uint8 HWC image for the preview window
    def imagination_to_rgb(self, imagination):
      if imagination is not None:
        x = imagination[0, 0]
      
        x = np.transpose(x, (1, 2, 0))
        x = (x + 0.5) * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

        return x
      
    #spawns the roof semantic camera (128x128, 120 fov, paper Sec. V-A), the collision and the lane invasion sensors
    def setupSensors(self):
      weak_self = weakref.ref(self)

      # ================== Camera Sensors ==================

      #setup semantic segmentation camera sensor
      camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
      camera_bp.set_attribute('image_size_x', f'128')
      camera_bp.set_attribute('image_size_y', f'128')
      camera_bp.set_attribute('fov', f'120')

      #spawn roof camera
      roofCameraPosition = carla.Transform(carla.Location(z = 1.6, x = 0.0, y = 0.0), carla.Rotation(pitch=30.0, yaw=0.0, roll=0.0))
      self.roofCamera = self.world.try_spawn_actor(camera_bp, roofCameraPosition, attach_to = self.vehicle)
      self.roofCamera.listen(lambda data: self._process_img(weak_self, data))
      self.sensor_list.append(self.roofCamera)

      # ================== Rest of Sensors ==================

      miscSensorsPosition = carla.Transform(carla.Location(z = 0.0, x = 0.0, y = 0.0))

      #spawn collision sensor
      collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
      self.collision_sensor = self.world.try_spawn_actor(collision_bp, miscSensorsPosition, attach_to = self.vehicle)
      self.collision_sensor.listen(lambda event: self._collision_data(weak_self, event))
      self.sensor_list.append(self.collision_sensor)

      #spawn lane invation sensor
      lane_invade_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
      self.lane_invade_sensor = self.world.try_spawn_actor(lane_invade_bp, miscSensorsPosition, attach_to = self.vehicle)
      self.lane_invade_sensor.listen(lambda event: self._lane_invade_data(weak_self, event))
      self.sensor_list.append(self.lane_invade_sensor)

    #camera callback: converts the raw frame to the CityScapes palette and stores it as the current observation
    @staticmethod
    def _process_img(weak_self, image):
      self = weak_self()
      if not self:
        return

      image.convert(carla.ColorConverter.CityScapesPalette)
      i = np.frombuffer(image.raw_data, dtype=np.uint8)
      i = i.reshape((128, 128, 4))[:, :, :3] #ignore 4th/alpha channel

      #set camera sensors to the processed observed image data
      self.roof_camera_observation = i

    #collision sensor callback, every event ends the episode in getReward
    @staticmethod
    def _collision_data(weak_self, event):
      self = weak_self()
      if not self:
        return

      self.collision_hist.append(event)

    #lane invasion sensor callback, solid line crossings end the episode, broken line crossings are only tracked
    @staticmethod
    def _lane_invade_data(weak_self, event):
      self = weak_self()
      if not self:
        return

      #determine what lane the vehicle invaded, allow to go throw broken lines
      lane_types = set(x.type for x in event.crossed_lane_markings)
      text = ['%r' % str(x).split()[-1] for x in lane_types]
      if text != ["'Broken'"]:
        self.lane_invade_hist.append(event)
      else:
        self.broken_lane_invade_hist.append(event) #we track broken lane invasion to make vahicle stay in one lane most of the time

    #starts a new episode: town change (random_towns), spawn the ego vehicle, plan the route, spawn the obstacle
    #of the selected regime, attach the sensors and return the first observation
    def reset(self, seed=None, options=None):
        #reset variables
        #self.close()
        
        #print("~~~~~~~~~~~~~~~~")
        #print("Reset Env")
        
        self.anomaly_detection = 0.0
        self.safety_brake = 0.0
        self.safety_throttle_scale = 1.0
        self.stopped_steps = 0
        self._current_step = 0
        self.waypoints_reached = 0
        self.distance_travelled = 0

        self.collision_hist.clear()
        self.lane_invade_hist.clear()
        self.broken_lane_invade_hist.clear()
        self.actor_list.clear()
        self.sensor_list.clear()

        #reset sensors
        self.roofCamera = None
        self.roof_camera_observation = None

        #reset vehicle
        self.vehicle = None
        self.npc_vehicle = None
        self.agent = None
        
        self.route.clear()
        
        self.initial_location = None
        self.next_point = None
        self.prev_point = None
        
        self.current_episode += 1
        
        #~~~~~~~~ town change (random_towns scenario) ~~~~~~~~
        if ROUTE_SCENARIO == "random_towns" and self.current_episode % TOWN_CHANGE_EVERY == 0:
            new_map = random.choice(self.map_list)
            self.client.load_world(new_map)
            self.world = self.client.get_world()
            #a freshly loaded world comes up asynchronous with the default delta, re-apply the synchronous 10 Hz settings
            #and refresh the handles that belong to the old world
            self.world.apply_settings(self.settings)
            self.map = self.world.get_map()
            self.spawn_points = self.map.get_spawn_points()
            self.spectator = self.world.get_spectator()
            self.vehicleBlueprint = self.world.get_blueprint_library().filter('model3')[0]
            print("Map Changed to:", new_map)
            
            time.sleep(0.1)
            self.world.tick()
        
        #~~~~~~~~ spawn point ~~~~~~~~
        if ROUTE_SCENARIO == "static":
            self.spawnPoint = carla.Transform(carla.Location(*STATIC_SPAWN_LOCATION), carla.Rotation(pitch=0.0, yaw=STATIC_SPAWN_YAW, roll=0.0))
        else:
            self.spawnPoint = random.choice(self.spawn_points)
        
        #print("~~~~~~~~~~~~~~~~")
        #print("Spawn Vehicle")
        #spawn new vehicle
        while self.vehicle is None:
          try:
            self.vehicle = self.world.try_spawn_actor(self.vehicleBlueprint, self.spawnPoint)
            #self.vehicle = self.world.spawn_actor(self.vehicleBlueprint, self.spawnPoint)
          except:
           pass

        self.actor_list.append(self.vehicle)
        self.agent = BasicAgent(self.vehicle)
        
        time.sleep(1)
        self.world.tick()
        
        route_attempts = 0
        while len(self.route) < MIN_ROUTE_WAYPOINTS:
          if route_attempts > 0 :
            if ROUTE_SCENARIO == "static":
              break #fixed route, keep whatever the planner returned
            self.spawnPoint = random.choice(self.spawn_points)
            self.vehicle.set_transform(self.spawnPoint)
            self.world.tick()
          self.get_waypoints()
          print("route length:", len(self.route))
          route_attempts += 1
        
        #~~~~~~~~ obstacle ~~~~~~~~
        self.spawn_npc_location = None
        self.npc_blueprint_id = None
        if OBSTACLE_REGIME != "none":
          if OBSTACLE_REGIME == "single":
            npc_blueprint = self.world.get_blueprint_library().find(SINGLE_OBSTACLE_BLUEPRINT)
          else:
            obstacles_bps = [bp for f in RANDOM_OBSTACLE_FILTERS for bp in self.world.get_blueprint_library().filter(f)]
            npc_blueprint = random.choice(obstacles_bps)
          
          #print("~~~~~~~~~~~~~~~~")
          #print("Spawn Obstacle")
          #spawn the obstacle on a random route point with a straight approach, pick a new point until it fits
          while self.npc_vehicle is None:
            random_point_index = self.pick_obstacle_index()
            self.spawn_npc_location = carla.Location(x=self.route[random_point_index][0], y=self.route[random_point_index][1], z=0.3)
            self.npc_spawnPoint = carla.Transform(self.spawn_npc_location)
            try:
              self.npc_vehicle = self.world.try_spawn_actor(npc_blueprint, self.npc_spawnPoint)
            except:
              pass
              
          self.actor_list.append(self.npc_vehicle)
          self.npc_blueprint_id = npc_blueprint.id
          #print("~~~~~~~~~~~~~~~~")
          #print("Obstacle Spawned")
          time.sleep(0.1)
          self.world.tick()

        #reset vehicle controls
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0, brake = 0.0))

        #spawn sensors
        self.setupSensors()

        while self.roofCamera is None:
          time.sleep(0.01)

        #print("~~~~~~~~~~~~~~~~")
        #print("Set initial rewards")          
        #INITIALIZE STATE
        terminated, truncated, reward = self.getReward()

        #print("~~~~~~~~~~~~~~~~")
        #print("Start new episode")
        return self.get_obs(), {}

    def render(self, close=False):
        return self._render_mode

    def seed(self, seed=None):
        pass

#the concrete environment: connects to the CARLA server, loads the town of the selected scenario,
#applies the synchronous 10 Hz settings and sets the action grid and reward weights (paper Sec. V-A)
class ContinuousCarEnv(CarEnv):
    #change this later
    SHOW_CAM = showPreview
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (128, 128, 3),
        n_steps: int = 500,
        action_dim: int = 2,
        dict_obs_space: bool = True,
    ):
        #discrete action space for dreamer
        self.discrete_acc = [-2.0, 0.0, 2.0] # discrete value of accelerations
        self.discrete_steer = [-0.6, -0.2, 0.0, 0.2, 0.6] # discrete value of steering angles
        
        self.n_steer = len(self.discrete_steer)
        self.n_acc = len(self.discrete_acc)

        self.action_space = gym.spaces.Discrete(self.n_steer * self.n_acc)

        # ~~~~~~~~~~ CAMERA SETUP PROPERTIES ~~~~~~~~~~ #
        #initialize sensors
        self.roofCamera = None

        # ~~~~~~~~~~ INITIALIZE SENSOR OBSERVATION ~~~~~~~~~~ #
        self.roof_camera_observation = None

        # ~~~~~~~~~~ INITIALIZE WORLD ~~~~~~~~~~ #
        #self.map_list = self.client.get_available_maps()
        self.map_list = TOWN_LIST
        
        #contect to carla client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(15.0)

        #load the world of the selected scenario
        if ROUTE_SCENARIO == "static":
            self.client.load_world(STATIC_TOWN)
        elif ROUTE_SCENARIO == "random":
            self.client.load_world(RANDOM_TOWN)
        else:
            self.client.load_world(random.choice(self.map_list))
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        #apply preview settings
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = not self.SHOW_CAM
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self.settings)

        # ~~~~~~~~~~ INITIALIZE ACTORS ~~~~~~~~~~ #
        self.actor_list = list()
        self.sensor_list = list()
        self.collision_hist = list()
        self.lane_invade_hist = list()
        self.broken_lane_invade_hist = list()
        self.route = list()
        
        self.spawn_points = self.map.get_spawn_points()
        self.next_point = None
        self.spawn_npc_location = None
        self.npc_blueprint_id = None
        
        # Setting up global router

        #set initial vehicle blueprint and position
        self.vehicleBlueprint = self.world.get_blueprint_library().filter('model3')[0]
        
        # ~~~~~~~~~~ SETUP REWARD WEIGHTS ~~~~~~~~~~ #
        self.total_timesteps = 500
        self.desired_speed = 8
        
        #anomaly score metric, change later to store the actual score
        self.waypoint_weight = 2.0
        self.speed_weight = 0.5
        self.collision_weight = 30.0
        self.lane_invasion_weight = 3.0
        self.time_weight = 0.1
        self.destination_reached_weight = 20.0

        # ~~~~~~~~~~ INITIALIZE INFO ~~~~~~~~~~ #
        self.state = {}
        self.terminal_conditions = {}
        self.obs_info = {}
        self.reward_info = {}

        # ~~~~~~~~~~ SETUP WORLD VIEW ~~~~~~~~~~ #
        if self.SHOW_CAM:
          self.spectator = self.world.get_spectator()

        super().__init__(
            image_size=image_size, n_steps=n_steps, dict_obs_space=dict_obs_space
        )
