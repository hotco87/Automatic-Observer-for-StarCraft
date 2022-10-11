from enum import Enum

TerranGroundUnits = [
    'Terran_Firebat',
    'Terran_Ghost',
    'Terran_Goliath',
    'Terran_Marine',
    'Terran_Medic',
    'Terran_SCV',
    'Terran_Siege_Tank_Siege_Mode',
    'Terran_Siege_Tank_Tank_Mde',
    'Terran_Vulture',
    'Terran_Vulture_Spider_Mine'
]

TerranAirUnits = [
    'Terran_Battlecruiser',
    'Terran_Dropship',
    'Terran_Nuclear_Missile',
    'Terran_Science_Vessel',
    'Terran_Valkyrie',
    'Terran_Wraith'
]

TerranBuildings = [
    'Terran_Academy',
    'Terran_Armory',
    'Terran_Barracks',
    'Terran_Bunker',
    'Terran_Command_Center',
    'Terran_Engineering_Bay',
    'Terran_Factory',
    'Terran_Missile_Turret',
    'Terran_Refinery',
    'Terran_Science_Facility',
    'Terran_Starport',
    'Terran_Supply_Depot'
]

TerranAddons = [
    'Terran_Comsat_Station',
    'Terran_Control_Tower',
    'Terran_Covert_Ops',
    'Terran_Machine_Shop',
    'Terran_Nuclear_Silo',
    'Terran_Physics_Lab',
]

ProtossGroundUnits = [
    'Protoss_Archon',
    'Protoss_Dark_Archon',
    'Protoss_Dark_Templar',
    'Protoss_Dragoon',
    'Protoss_High_Templar',
    'Protoss_Probe',
    'Protoss_Reaver',
    'Protoss_Scarab',
    'Protoss_Zealot'
]

ProtossAirUnits = [
    'Protoss_Arbiter',
    'Protoss_Carrier',
    'Protoss_Corsair',
    'Protoss_Interceptor',
    'Protoss_Observer',
    'Protoss_Scout',
    'Protoss_Shuttle'
]

ProtossBuildings = [
    'Protoss_Arbiter_Tribunal',
    'Protoss_Assimilator',
    'Protoss_Citadel_of_Adun',
    'Protoss_Cybernetics_Core',
    'Protoss_Fleet_Beacon',
    'Protoss_Forge',
    'Protoss_Gateway',
    'Protoss_Nexus',
    'Protoss_Observatory',
    'Protoss_Photon_Cannon',
    'Protoss_Pylon',
    'Protoss_Robotics_Facility',
    'Protoss_Robotics_Support_Bay',
    'Protoss_Shield_Battery',
    'Protoss_Stargate',
    'Protoss_Templar_Archives'
]

ZergGroundUnits = [
    'Zerg_Broodling',
    'Zerg_Defiler',
    'Zerg_Drone',
    'Zerg_Egg',
    'Zerg_Hydralisk',
    'Zerg_Infested_Terran',
    'Zerg_Larva',
    'Zerg_Lurker',
    'Zerg_Lurker_Egg',
    'Zerg_Ultralisk',
    'Zerg_Zergling'
]

ZergAirUnits = [
    'Zerg_Cocoon',
    'Zerg_Devourer',
    'Zerg_Guardian',
    'Zerg_Mutalisk',
    'Zerg_Overlord',
    'Zerg_Queen',
    'Zerg_Scourge'
]

ZergBuildings = [
    'Zerg_Creep_Colony',
    'Zerg_Defiler_Mound',
    'Zerg_Evolution_Chamber',
    'Zerg_Extractor',
    'Zerg_Greater_Spire',
    'Zerg_Hatchery',
    'Zerg_Hive',
    'Zerg_Hydralisk_Den',
    'Zerg_Infested_Command_Center',
    'Zerg_Lair',
    'Zerg_Nydus_Canal',
    'Zerg_Queens_Nest',
    'Zerg_Spawning_Pool',
    'Zerg_Spire',
    'Zerg_Spore_Colony',
    'Zerg_Sunken_Colony',
    'Zerg_Ultralisk_Cavern'
]

Critters = [
    'Critter_Bengalaas',
    'Critter_Kakaru',
    'Critter_Ragnasaur',
    'Critter_Rhynadon',
    'Critter_Scantid',
    'Critter_Ursadon'
]

Resources = [
    'Resource_Mineral_Field',
    'Resource_Mineral_Field_Type_2',
    'Resource_Mineral_Field_Type_3',
    'Resource_Vespene_Geyser'
]

Spells = [
    'Spell_Dark_Swarm',
    'Spell_Disruption_Web',
    'Spell_Scanner_Sweep'
]


class UnitType():
    def __init__(self, name: str):
        self._name = name

    @property
    def isWorker(self):
        return self._name in ['Terran_SCV', 'Protoss_Probe', 'Zerg_Drone']

    @property
    def isUnit(self):
        return self._name in TerranGroundUnits or \
               self._name in TerranAirUnits or \
               self._name in ProtossGroundUnits or \
               self._name in ProtossAirUnits or \
               self._name in ZergGroundUnits or \
               self._name in ZergAirUnits or \
               self._name in Critters

    @property
    def isGround(self):
        return self._name in TerranGroundUnits or \
               self._name in ProtossGroundUnits or \
               self._name in ZergGroundUnits

    @property
    def isAir(self):
        return self._name in TerranAirUnits or \
               self._name in ProtossAirUnits or \
               self._name in ZergAirUnits

    @property
    def isBuilding(self):
        return self._name in TerranBuildings or self._name in TerranAddons or \
               self._name in ProtossBuildings or \
               self._name in ZergBuildings

    @property
    def isAddon(self):
        return self._name in TerranAddons

    @property
    def isTrivial(self):
        return self._name in Critters or \
               self._name in ['Zerg_Larva']

    @property
    def isSpell(self):
        return self._name in Spells

    @property
    def isResource(self):
        return self._name in Resources
