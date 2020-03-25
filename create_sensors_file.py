from pyFormulaClientNoNvidia import messages
from pyFormulaClientNoNvidia.FormulaClient import FormulaClient, ClientSource, SYSTEM_RUNNER_IPC_PORT

import os

def main(): 
    client = FormulaClient(ClientSource.SERVER, 
        read_from_file=os.devnull, write_to_file='sensors.messages')
    conn = client.connect(SYSTEM_RUNNER_IPC_PORT)
    
    # Create camera data
    camera_data = messages.sensors.CameraSensor()
    camera_data.width = 2
    camera_data.height = 1
    camera_data.pixels = b'\x01\x02'

    # Create the message wrapper and save to file
    camera_msg = messages.common.Message()
    camera_msg.data.Pack(camera_data)
    conn.send_message(camera_msg)

    exit_data = messages.server.ExitMessage()
    exit_msg = messages.common.Message()
    exit_msg.data.Pack(exit_data)
    conn.send_message(exit_msg)


if __name__ == '__main__':
    main()
