from PerceptionClient import PerceptionClient
from pyFormulaClientNoNvidia import messages
from perception_functions import get_cones_from_camera
import time
import signal
import sys
import math

class Perception:
    def __init__(self):
        # The sensors.messages can be created using the create_sensors_file.py
        # PerceptionClient(<path to read messages from>, <path to write sent messages to>)
        self._client = PerceptionClient('sensors.messages', 'perception.messages')
        self._running_id = 1
        self.message_timeout = 0.01

    def start(self):
        self._client.connect(1)
        self._client.set_read_delay(0.05) # Sets the delay between reading new messages from sensors.messages
        self._client.start()

    def stop(self):
        if self._client.is_alive():
            self._client.stop()
            self._client.join()
 
    def process_camera_message(self, camera_msg):
        camera_data = messages.sensors.CameraSensor()
        camera_msg.data.Unpack(camera_data)
        # Camera data has the following properties: width, height, pixels
        print(f"Got camera width: {camera_data.width}, height: {camera_data.height}")

        # Create the new cone map and append to it all of the recognized cones from the image
        cone_map = messages.perception.ConeMap()

        # get image cones: img_cones=[[x,y,w,h,type,depth],[x,y,w,h,type,depth],....]
        # x,y - left top bounding box position in image plain
        # w, h - width and height of bounding box in pixels
        # type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
        # depth - nominal depth value

        img_cones = get_cones_from_camera(camera_data.width, camera_data.height, camera_data.pixels)
        


        # transformation from image plain to cartesian coordinate system
        boxes = [
            {
                'id': 1,
                'x': 25,
                'y': 1,
                'z': 1,
                'type': messages.perception.Blue
            }
        ]

        for box in boxes:  
            #   Create new cone and set its properties
            cone = messages.perception.Cone()
            cone.cone_id = box['id']
            cone.type = box['type']
            cone.x = box['x']
            cone.y = box['y']
            cone.z = box['z']
            cone_map.cones.append(cone) # apped the new cone to the cone map

        return cone_map

    def process_server_message(self, server_messages):
        if server_messages.data.Is(messages.server.ExitMessage.DESCRIPTOR):
            return True

        return False

    def send_message2state(self, msg_id, cone_map):
        msg = messages.common.Message()
        msg.header.id = msg_id
        msg.data.Pack(cone_map)
        self._client.send_message(msg)

    def run(self):
        while True:
            try:
                server_msg = self._client.pop_server_message()
                if server_msg is not None:
                    if self.process_server_message(server_msg):
                        return
            except Exception as e:
                pass
                
            try:
                # In the future can be replaced with getting the camera directly
                camera_msg = self._client.get_camera_message(timeout=self.message_timeout) 
                cone_map = self.process_camera_message(camera_msg)
                self.send_message2state(camera_msg.header.id, cone_map)    
            except Exception as e:
                #pass
                print(e)

perception = Perception()

def stop_all_threads():
    print("Stopping threads")
    perception.stop()

def shutdown(a, b):
    print("Shutdown was called")
    stop_all_threads()
    exit(0)


def main():
    print("Initalized Perception")

    perception.start()
    perception.run()

    stop_all_threads()
    exit(0)

if __name__ == "__main__":
    for signame in ('SIGINT', 'SIGTERM'):
        signal.signal(getattr(signal, signame), shutdown)
    main()