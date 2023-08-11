# Brick-vocab-mapper

```python
from mapper import Mapper
from rdflib import BRICK
import urllib.request
import csv

defns = []
# read haystack proto definitions off their website
request = urllib.request.Request("https://project-haystack.org/download/protos.csv", headers={'User-Agent': "Python script"})
response = urllib.request.urlopen(request)
rdr = csv.DictReader(response.read().decode('utf8').splitlines())
for row in rdr:
    # save each proto (e.g. 'air temp sensor point') as a dictionary
    defns.append({'name': row['proto']})

# create a new Mapper object which knows about the definitions. This will
# access OpenAI to compute embeddings if necessary
m = Mapper(defns, "haystack_proto.index")
# define which set of Brick classes we want to match against
# all points which aren't parameters (these have similar enough names it can confuse the mapping)
points = m.get_brick_classes(BRICK.Point, [BRICK.Parameter])
# all equipment in Brick
equips = m.get_brick_classes(BRICK.Equipment)
# concatenate these two lists of Brick classes
bcs = points + equips

# compute a 'stable' mapping (1:1 mapping)
stable = m.get_mapping(bcs, allow_collisions=False)
# compute an 'unstable' mapping that allows 1:n and n:1 matches
unstable = m.get_mapping(bcs, allow_collisions=True)

# loop through and print the stable/unstable matches
for k,v in unstable.items():
    print(f"{k}:\n\tunstable: {v}\n\tstable: {stable.get(k)}")
```

Gives output like:

```
flue hot water heat valve cmd point:
        unstable: https://brickschema.org/schema/Brick#Domestic_Hot_Water_Valve
        stable: https://brickschema.org/schema/Brick#Hot_Water_Valve
steam bypass pump point:
        unstable: https://brickschema.org/schema/Brick#Bypass_Valve
        stable: https://brickschema.org/schema/Brick#Bypass_Valve
makeup water entering valve actuator equip:
        unstable: https://brickschema.org/schema/Brick#Makeup_Water_Valve
        stable: https://brickschema.org/schema/Brick#Makeup_Water_Valve
makeup water entering pipe equip:
        unstable: https://brickschema.org/schema/Brick#Eye_Wash_Station
        stable: https://brickschema.org/schema/Brick#Eye_Wash_Station
steam leaving pump point:
        unstable: https://brickschema.org/schema/Brick#Emergency_Wash_Station
        stable: https://brickschema.org/schema/Brick#Steam_Baseboard_Radiator
infiltration well equip:
        unstable: https://brickschema.org/schema/Brick#Intrusion_Detection_Equipment
        stable: https://brickschema.org/schema/Brick#Intrusion_Detection_Equipment
economizer damper actuator equip:
        unstable: https://brickschema.org/schema/Brick#Economizer_Damper
        stable: https://brickschema.org/schema/Brick#Economizer_Damper
steam leaving point:
        unstable: https://brickschema.org/schema/Brick#Steam_Distribution
        stable: https://brickschema.org/schema/Brick#Steam_Distribution
flue equip:
        unstable: https://brickschema.org/schema/Brick#Fire_Safety_Equipment
        stable: https://brickschema.org/schema/Brick#Fire_Safety_Equipment
condenser water entering pump equip:
        unstable: https://brickschema.org/schema/Brick#Condenser_Water_Pump
        stable: https://brickschema.org/schema/Brick#Condenser_Water_Pump
```

### UI notes:
- is mapping correct/incorrect
