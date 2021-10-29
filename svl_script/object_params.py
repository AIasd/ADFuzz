class Waypoint:
    def __init__(self, trigger_distance, x, y):
        self.trigger_distance = trigger_distance
        self.x = x
        self.y = y

class Object:
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

class Static(Object):
    def __init__(self, model, x, y):
        super().__init__(model, x, y)

class Pedestrian(Object):
    def __init__(self, model, x, y, speed, waypoints):
        super().__init__(model, x, y)
        self.speed = speed
        self.waypoints = waypoints


class Vehicle(Object):
    def __init__(self, model, x, y, speed, waypoints):
        super().__init__(model, x, y)
        self.speed = speed
        self.waypoints = waypoints
