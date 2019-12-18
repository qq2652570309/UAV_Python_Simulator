class Drone:
    def __init__(self, id=-1, launch_row=-1, launch_col=-1, landing_row=-1, landing_col=-1):
        self.id = id
        self.current_row = launch_row
        self.current_col = launch_col
        self.landing_row = landing_row
        self.landing_col = landing_col