from abc import ABC, abstractmethod
import math

from typing import Tuple


class VisualObject(ABC):
    @property
    @abstractmethod
    def type(self):
        return

    @property
    @abstractmethod
    def alpha(self):
        pass

    @property
    @abstractmethod
    def rectangle(self):
        pass

    @property
    @abstractmethod
    def box(self):
        pass

    @property
    @abstractmethod
    def rotation(self):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @property
    @abstractmethod
    def is_horizontal(self):
        pass

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @property
    @abstractmethod
    def content(self):
        pass


class Rectangle(ABC):
    @abstractmethod
    def abs_size(self):
        pass

    @abstractmethod
    def abs_origin(self):
        pass

    @abstractmethod
    def abs_rotation(self):
        pass

    @abstractmethod
    def is_horizontal(self):
        pass


class RectangleForm(Rectangle):
    HORIZONTAL_THRESHOLD = 0.01

    def __init__(self, size: Tuple[float, float], origin: Tuple[float, float], rotation: float, parent: Rectangle):
        self._origin = origin
        self._size = size
        self._rotation = rotation
        self._parent = parent

    def abs_size(self):
        parent_size = self._parent.abs_size()
        return parent_size[0] * self._size[0], parent_size[1] * self._size[1]

    def abs_origin(self):
        parent_origin = self._parent.abs_origin()
        parent_rotation = self._parent.abs_rotation()
        return parent_origin[0] + self._origin[0] * math.cos(parent_rotation[0]),\
               parent_origin[1] + self._origin[1] * math.sin(parent_rotation[1])

    def abs_rotation(self):
        return self._rotation + self._parent.abs_rotation()

    def is_horizontal(self):
        return abs(self._rotation) < self.HORIZONTAL_THRESHOLD


class MainFrame(RectangleForm):
    def __init__(self, size: Tuple[float, float], origin=(0, 0), rotation=0., parent=None):
        super(MainFrame, self).__init__(size, origin, rotation, parent)

    def abs_size(self):
        return self._size

    def abs_origin(self):
        return self._origin

    def abs_rotation(self):
        return self._rotation

    def is_horizontal(self):
        return True

