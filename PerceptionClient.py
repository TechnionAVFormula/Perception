from config import CONFIG, IN_MESSAGE_FILE, OUT_MESSAGE_FILE
from config import ConfigEnum

if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
    from pyFormulaClient import FormulaClient, messages  
    from pyFormulaClient.ModuleClient import ModuleClient
    from pyFormulaClient.MessageDeque import MessageDeque
elif ( CONFIG == ConfigEnum.LOCAL_TEST):
    from pyFormulaClientNoNvidia import FormulaClient, messages  
    from pyFormulaClientNoNvidia.ModuleClient import ModuleClient
    from pyFormulaClientNoNvidia.MessageDeque import MessageDeque
else:
    raise NameError('User Should Choose Configuration from config.py')


class PerceptionClient(ModuleClient):
    def __init__(self, read_from_file, write_to_file):
        if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
            super().__init__(FormulaClient.ClientSource.STATE_EST)       
        elif ( CONFIG == ConfigEnum.LOCAL_TEST):
            super().__init__(FormulaClient.ClientSource.STATE_EST, IN_MESSAGE_FILE, OUT_MESSAGE_FILE)  
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
