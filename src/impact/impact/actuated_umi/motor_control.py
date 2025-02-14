from dynamixel_api import Motor


class ActuatedUMI(Motor):

    def __init__(self, connector):
        super().__init__(connector)

    @property
    def position_p_gain(self):
        return self._Motor__read("position_p_gain")
    
    @position_p_gain.setter
    def position_p_gain(self, value: int):
        self._Motor__write("position_p_gain", value)

    @property
    def position_i_gain(self):
        return self._Motor__read("position_i_gain")
    
    @position_i_gain.setter
    def position_i_gain(self, value: int):
        self._Motor__write("position_i_gain", value)

    @property
    def position_d_gain(self):
        return self._Motor__read("position_d_gain")
    
    @position_d_gain.setter
    def position_d_gain(self, value: int):
        self._Motor__write("position_d_gain", value)

    @property
    def current_load(self):
        return self._Motor__read("present_load")