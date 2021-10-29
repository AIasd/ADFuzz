import carla


WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,



        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.CloudySunset,



        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset,



        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.MidRainSunset,



        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.WetCloudySunset,



        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.HardRainSunset,



        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,


        # ClearNight
        carla.WeatherParameters(15.0, 0.0, 0.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # CloudyNight
        carla.WeatherParameters(80.0, 0.0, 0.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # WetNight
        carla.WeatherParameters(20.0, 0.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # MidRainNight
        carla.WeatherParameters(90.0, 0.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # WetCloudyNight
        carla.WeatherParameters(80.0, 30.0, 50.0, 0.40, 0.0, -90.0, 0.0, 0.0, 0.0),

        # HardRainNight
        carla.WeatherParameters(80.0, 60.0, 100.0, 1.00, 0.0, -90.0, 0.0, 0.0, 0.0),

        # SoftRainNight
        carla.WeatherParameters(90.0, 15.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),
]

weather_names = ['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset', 'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset', 'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon', 'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset', 'ClearNight', 'CloudyNight', 'WetNight', 'MidRainNight', 'WetCloudyNight', 'HardRainNight', 'SoftRainNight']




# WEATHERS = [
#         carla.WeatherParameters.ClearNoon,
#         carla.WeatherParameters.ClearSunset,
#
#
#
#         carla.WeatherParameters.CloudyNoon,
#         carla.WeatherParameters.CloudySunset,
#
#
#
#         carla.WeatherParameters.WetNoon,
#         carla.WeatherParameters.WetSunset,
#
#
#
#         carla.WeatherParameters.MidRainyNoon,
#         carla.WeatherParameters.MidRainSunset,
#
#
#
#         carla.WeatherParameters.WetCloudyNoon,
#         carla.WeatherParameters.WetCloudySunset,
#
#
#
#         carla.WeatherParameters.HardRainNoon,
#         carla.WeatherParameters.HardRainSunset,
#
#
#
#         carla.WeatherParameters.SoftRainNoon,
#         carla.WeatherParameters.SoftRainSunset,
#
#
# ]
#
# weather_names = ['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset', 'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset', 'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon', 'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset']


# walker modifiable attributes: speed: float
pedestrian_types = ['walker.pedestrian.00'+f'{i:02d}' for i in range(1, 14)]



# vehicle types
# car
car_types = ['vehicle.audi.a2',
'vehicle.audi.tt',
'vehicle.mercedes-benz.coupe',
'vehicle.bmw.grandtourer',
'vehicle.audi.etron',
'vehicle.nissan.micra',
'vehicle.lincoln.mkz2017',
'vehicle.tesla.cybertruck',
'vehicle.dodge_charger.police',
'vehicle.tesla.model3',
'vehicle.toyota.prius',
'vehicle.seat.leon',
'vehicle.nissan.patrol',
'vehicle.mini.cooperst',
'vehicle.jeep.wrangler_rubicon',
'vehicle.mustang.mustang',
'vehicle.volkswagen.t2',
'vehicle.chevrolet.impala',
'vehicle.citroen.c3']

large_car_types = ['vehicle.carlamotors.carlacola']

# motorcycle
motorcycle_types = ['vehicle.yamaha.yzf',
'vehicle.harley-davidson.low_rider',
'vehicle.kawasaki.ninja']

# cyclist
cyclist_types = ['vehicle.bh.crossbike',
'vehicle.gazelle.omafiets',
'vehicle.diamondback.century']

vehicle_types = car_types + large_car_types + motorcycle_types + cyclist_types


# static objects
static_types = ['static.prop.shoppingtrolley',
# 'static.prop.dirtdebris01',
'static.prop.barrel',
'static.prop.table',
# 'static.prop.box02',
'static.prop.trashcan04',
# 'static.prop.garbage03',
# 'static.prop.plantpot02',
'static.prop.streetbarrier',
# 'static.prop.plantpot01',
# 'static.prop.briefcase',
'static.prop.clothesline',
'static.prop.plasticchair',
# 'static.prop.trampoline',
# 'static.prop.motorhelmet',
# 'static.prop.chainbarrierend',
# 'static.prop.creasedbox01',
# 'static.prop.kiosk_01',
# 'static.prop.clothcontainer',
'static.prop.barbeque',
'static.prop.streetsign01',
'static.prop.bench01',
# 'static.prop.platformgarbage01',
# 'static.prop.garbage06',
'static.prop.bench03',
# 'static.prop.bin',
# 'static.prop.purse',
# 'static.prop.box01',
# 'static.prop.brokentile01',
# 'static.prop.wateringcan',
'static.prop.plastictable',
# 'static.prop.slide',
# 'static.prop.dirtdebris03',
# 'static.prop.creasedbox02',
# 'static.prop.brokentile02',
'static.prop.streetsign',
# 'static.prop.garbage04',
# 'static.prop.garbage02',
# 'static.prop.pergola',
'static.prop.constructioncone',
# 'static.prop.advertisement',
# 'static.prop.plantpot04',
# 'static.prop.bikeparking',
# 'static.prop.creasedbox03',
# 'static.prop.trashbag',
# 'static.prop.plasticbag',
'static.prop.trashcan05',
# 'static.prop.busstop',
# 'static.prop.trafficcone02',
# 'static.prop.travelcase',
'static.prop.trashcan02',
'static.prop.streetfountain',
# 'static.prop.dirtdebris02',
# 'static.prop.atm',
# 'static.prop.shop01',
'static.prop.shoppingcart',
# 'static.prop.doghouse',
# 'static.prop.trafficcone01',
# 'static.prop.garbage01',
'static.prop.swingcouch',
# 'static.prop.bike helmet',
# 'static.prop.brokentile04',
'static.prop.trashcan03',
# 'static.prop.plantpot07',
# 'static.prop.plantpot05',
# 'static.prop.mobile',
# 'static.prop.trashcan01',
'static.prop.maptable',
# 'static.prop.guitarcase',
'static.prop.trafficwarning',
# 'static.prop.shoppingbag',
# 'static.prop.mailbox',
# 'static.prop.container',
# 'static.prop.plantpot03',
# 'static.prop.plantpot06',
# 'static.prop.glasscontainer',
# 'static.prop.plantpot08',
'static.prop.streetsign04',
'static.prop.gardenlamp',
'static.prop.bench02',
# 'static.prop.box03',
# 'static.prop.swing',
# 'static.prop.garbage05',
# 'static.prop.ironplank',
# 'static.prop.gnome',
# 'static.prop.fountain',
# 'static.prop.colacan',
# 'static.prop.chainbarrier',
# 'static.prop.vendingmachine',
# 'static.prop.brokentile03',
]


# vehicle colors
# black, white, gray, silver, blue, red, brown, gold, green, tan, orange
vehicle_colors = ['(0, 0, 0)',
'(255, 255, 255)',
'(220, 220, 220)',
'(192, 192, 192)',
'(0, 0, 255)',
'(255, 0, 0)',
'(165,42,42)',
'(255,223,0)',
'(0,128,0)',
'(210,180,140)',
'(255,165,0)']

















'''
When using smaller number of choices
'''
WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,



        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.CloudySunset,



        # carla.WeatherParameters.WetNoon,
        # carla.WeatherParameters.WetSunset,



        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.MidRainSunset,



        # carla.WeatherParameters.WetCloudyNoon,
        # carla.WeatherParameters.WetCloudySunset,



        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.HardRainSunset,



        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,


        # ClearNight
        carla.WeatherParameters(15.0, 0.0, 0.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # CloudyNight
        carla.WeatherParameters(80.0, 0.0, 0.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # # WetNight
        # carla.WeatherParameters(20.0, 0.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # MidRainNight
        carla.WeatherParameters(90.0, 0.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),

        # # WetCloudyNight
        # carla.WeatherParameters(80.0, 30.0, 50.0, 0.40, 0.0, -90.0, 0.0, 0.0, 0.0),

        # HardRainNight
        carla.WeatherParameters(80.0, 60.0, 100.0, 1.00, 0.0, -90.0, 0.0, 0.0, 0.0),

        # SoftRainNight
        carla.WeatherParameters(90.0, 15.0, 50.0, 0.35, 0.0, -90.0, 0.0, 0.0, 0.0),
]

weather_names = ['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset',  'MidRainyNoon', 'MidRainSunset',  'HardRainNoon', 'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset', 'ClearNight', 'CloudyNight', 'MidRainNight', 'HardRainNight', 'SoftRainNight']

assert len(WEATHERS) == len(weather_names)

# walker modifiable attributes: speed: float
pedestrian_types = ['walker.pedestrian.00'+f'{i:02d}' for i in [1, 5, 13, 14]]



# vehicle types
# car
car_types = [
'vehicle.tesla.cybertruck',
'vehicle.dodge_charger.police',
'vehicle.tesla.model3',
'vehicle.mini.cooperst',
'vehicle.jeep.wrangler_rubicon',
'vehicle.volkswagen.t2']

large_car_types = ['vehicle.carlamotors.carlacola']

# motorcycle
motorcycle_types = ['vehicle.yamaha.yzf',
'vehicle.harley-davidson.low_rider']

# cyclist
cyclist_types = ['vehicle.bh.crossbike',
'vehicle.gazelle.omafiets']

vehicle_types = car_types + large_car_types + motorcycle_types + cyclist_types



vehicle_colors = ['(0, 0, 0)',
'(255, 255, 255)',
'(220, 220, 220)',
'(192, 192, 192)',
'(0, 0, 255)']
