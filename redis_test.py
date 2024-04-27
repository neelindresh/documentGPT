from dataclasses import dataclass,asdict

@dataclass
class RedisBroker:
    host:str="localhost"
    port:int=6379
    username:str="default"
    password:str="admin"
    db:int=0
