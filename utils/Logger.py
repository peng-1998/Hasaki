import os
import json
import time
from typing import Dict


class Logger:
    def __init__(self, save_path: str):
        if os.path.exists(save_path):
            self.alllog = json.load(open(save_path))
            assert isinstance(self.alllog, list)
        else:
            self.alllog = []
        self.log = Dict()
        self.save_path = save_path
        self.alllog.append(self.log)

    def step(self, log: Dict):
        self.log['time'] = time.asctime(time.localtime(time.time()))
        for key in log.keys():
            if key in self.log.keys():
                self.log[key].append(log[key])
            else:
                self.log[key] = [log[key]]
        json.dump(self.alllog, open(self.save_path, 'w'))
    
    def addlog(self,key:str,value:str):
        self.log[key] = value
