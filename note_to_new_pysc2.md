## a note to the newest version(1.2) of pysc2

### avaiable action:
obs.observation.available_action

### timestep class:
`['step_type', 'reward', 'discount', 'observation']`

      A `TimeStep` namedtuple containing:
        step_type: A `StepType` of `FIRST`.
        reward: `None`, indicating the reward is undefined.
        discount: `None`, indicating the discount is undefined.
        observation: A NumPy array, or a dict, list or tuple of arrays
          corresponding to `observation_spec()`.
          
### def parse_agent_interface_format():
this function wrap the ground level layers to create the environment in bin/agent.py for the agent to play 

### obs.observation.screen_feature:
it's of size (17 = # features, 64, 64 = screen size)

### screen features:
These are the screen feature layers:

height_map: Shows the terrain levels.

visibility: Which part of the map are hidden, have been seen or are currently visible.

creep: Which parts have zerg creep.

power: Which parts have protoss power, only shows your power.

player_id: Who owns the units, with absolute ids.

player_relative: Which units are friendly vs hostile. Takes values in [0, 4], denoting [background, self, ally, neutral, enemy] units respectively.

unit_type: A unit type id, which can be looked up in pysc2/lib/units.py.

selected: Which units are selected.

hit_points: How many hit points the unit has.

energy: How much energy the unit has.

shields: How much shields the unit has. Only for protoss units.

unit_density: How many units are in this pixel.

unit_density_aa: An anti-aliased version of unit_density with a maximum of 16 per unit per pixel. This gives you sub-pixel unit location and size. For example if a unit is exactly 1 pixel diameter, unit_density will show it in exactly 1 pixel regardless of where in that pixel it is actually centered. unit_density_aa will instead tell you how much of each pixel is covered by the unit. A unit that is smaller than a pixel and centered in the pixel will give a value less than the max. A unit with diameter 1 centered near the corner of a pixel will give roughly a quarter of its value to each of the 4 pixels it covers. If multiple units cover a pixel their proportion of the pixel covered will be summed, up to a max of 256.