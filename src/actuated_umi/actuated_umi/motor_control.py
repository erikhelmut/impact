from dynamixel_api import Motor


class ActuatedUMI(Motor):

    def __init__(self, connector):
        super().__init__(connector)

    def __del__(self):
        self.connector.write_field("torque_enable", False)

    @property
    def position_p_gain(self):
        return self.connector.read_field("position_p_gain")
    
    @position_p_gain.setter
    def position_p_gain(self, value: int):
        self.connector.write_field("position_p_gain", value)

    @property
    def position_i_gain(self):
        return self.connector.read_field("position_i_gain")
    
    @position_i_gain.setter
    def position_i_gain(self, value: int):
        self.connector.write_field("position_i_gain", value)

    @property
    def position_d_gain(self):
        return self.connector.read_field("position_d_gain")
    
    @position_d_gain.setter
    def position_d_gain(self, value: int):
        self.connector.write_field("position_d_gain", value)

    @property
    def velocity_p_gain(self):
        return self.connector.read_field("velocity_p_gain")
    
    @velocity_p_gain.setter
    def velocity_p_gain(self, value: int):
        self.connector.write_field("velocity_p_gain", value)

    @property
    def velocity_i_gain(self):
        return self.connector.read_field("velocity_i_gain")
    
    @velocity_i_gain.setter
    def velocity_i_gain(self, value: int):
        self.connector.write_field("velocity_i_gain", value)

    @property
    def current_load(self):
        return self.connector.read_field("present_load")