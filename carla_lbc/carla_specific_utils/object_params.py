class Object:
    def __init__(self, model, spawn_transform):
        self.model = model
        self.spawn_transform = spawn_transform
class Static(Object):
    def __init__(self, model, spawn_transform):
        super().__init__(model, spawn_transform)
class Pedestrian(Object):
    def __init__(self, model, spawn_transform, trigger_distance, speed, dist_to_travel, after_trigger_behavior):
        super().__init__(model, spawn_transform)
        self.trigger_distance = trigger_distance
        self.speed = speed
        self.dist_to_travel = dist_to_travel
        # ['destroy', 'stop']
        self.after_trigger_behavior = after_trigger_behavior
class Vehicle(Object):
    def __init__(self, model, spawn_transform, avoid_collision, initial_speed, trigger_distance, waypoint_follower, targeted_waypoint, dist_to_travel, target_direction, targeted_speed, after_trigger_behavior, color, waypoints_perturbation):
        super().__init__(model, spawn_transform)
        self.initial_speed = initial_speed
        self.trigger_distance = trigger_distance
        self.targeted_speed = targeted_speed

        # Boolean
        self.waypoint_follower = waypoint_follower
        # If True
        self.targeted_waypoint = targeted_waypoint
        self.avoid_collision = avoid_collision
        # If False
        self.dist_to_travel = dist_to_travel
        self.target_direction = target_direction

        # ['destroy', 'stop']
        self.after_trigger_behavior = after_trigger_behavior
        self.color = color

        # this field won't be used unless the model name is unknown
        self.category = 'car'


        self.waypoints_perturbation = waypoints_perturbation
