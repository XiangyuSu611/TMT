import logging

logger = logging.getLogger(__name__)


class Light:
    pass


class PointLight(Light):
    type = 0

    def __init__(self, position, intensity, color=(1.0, 1.0, 1.0)):
        self.position = position
        self.intensity = intensity
        self.color = color


class DirectionalLight(Light):
    type = 1

    def __init__(self, direction, intensity, color=(1.0, 1.0, 1.0)):
        self.position = direction
        self.intensity = intensity
        self.color = color
