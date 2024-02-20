

class Units:
    def __init__(self,magnitude:float,unit:str,prefix:str) -> None:
        self.magnitude = magnitude
        self.unit = unit




def to_inch(x:float, unit:str)-> float:
    if unit == "px":
        return x/64
    


