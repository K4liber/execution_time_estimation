import json


class DataDetails:
    # Size unit is a byte
    def __init__(self, overall_size: int, parts: int, element_avg_size: int, element_max_size: int):
        self._overall_size = overall_size
        self._parts = parts
        self._element_avg_size = element_avg_size
        self._element_max_size = element_max_size

    @property
    def overall_size(self) -> int:
        return self._overall_size

    @property
    def parts(self) -> int:
        return self._parts

    @property
    def element_avg_size(self) -> int:
        return self._element_avg_size

    @property
    def element_max_size(self) -> int:
        return self._element_max_size

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=lambda o: o.__dict__, indent=3)

    def __repr__(self) -> str:
        return self.to_json()
