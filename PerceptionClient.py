from pyFormulaClientNoNvidia.ModuleClient import ModuleClient, ClientSource
from pyFormulaClientNoNvidia.MessageDeque import MessageDeque
from pyFormulaClientNoNvidia import messages

class PerceptionClient(ModuleClient):
    def __init__(self, read_from_file, write_to_file):
        super().__init__(ClientSource.PERCEPTION, read_from_file, write_to_file)    
        self.server_messages = MessageDeque()                                              
        self.camera_messages = MessageDeque(maxlen=1)        

    def _callback(self, msg):  
        if msg.data.Is(messages.sensors.CameraSensor.DESCRIPTOR):
            self.camera_messages.put(msg)
        else:
            self.server_messages.put(msg)

    def get_camera_message(self, blocking=True, timeout=None):
        return self.camera_messages.get(blocking, timeout)

    def pop_server_message(self, blocking=False, timeout=None):
        return self.server_messages.get(blocking, timeout)
