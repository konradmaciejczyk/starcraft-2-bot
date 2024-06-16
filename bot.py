import sc2 
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import COMMANDCENTER, SCV, SUPPLYDEPOT, REFINERY, \
                          BARRACKS, FACTORY, STARPORT, FACTORYTECHLAB, \
                          FACTORYREACTOR, BARRACKSREACTOR, BARRACKSTECHLAB, \
                          MARINE, HELLION
from sc2.ids.unit_typeid import UnitTypeId
import random

class TerranCustomBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50
        self.MAX_BARRACKS = 5
        self.MAX_FACTORIES = 5
        self.MAX_STARPORTS = 5
        self.MAX_MISSILETURRETS = 10

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.train_workers()
        await self.build_supply_depots()
        await self.build_refinery()
        await self.expand()
        await self.offensive_force_buildings()
        # await self.buildings_attachments()
        await self.train_offensive_force()
        await self.attack()
        # await self.build_defenses()

    async def train_workers(self):
        if (len(self.units(COMMANDCENTER)) * 16) > len(self.units(SCV)) and len(self.units(SCV)) < self.MAX_WORKERS:
            for commandcenter in self.units(COMMANDCENTER).ready.noqueue:
                if self.can_afford(SCV):
                    await self.do(commandcenter.train(SCV))

    async def build_supply_depots(self):
        if self.supply_left < 5 and not self.already_pending(SUPPLYDEPOT):
            commandcenters = self.units(COMMANDCENTER).ready
            if commandcenters.exists:
                if self.can_afford(SUPPLYDEPOT):
                    await self.build(SUPPLYDEPOT, near=commandcenters.first)

    async def build_refinery(self):
        for commandcenter in self.units(COMMANDCENTER).ready:
            vaspenes = self.state.vespene_geyser.closer_than(20.0, commandcenter)
            for vaspene in vaspenes:
                if not self.can_afford(REFINERY):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(REFINERY).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(REFINERY, vaspene))

    async def expand(self):
        if self.units(COMMANDCENTER).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(COMMANDCENTER):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(SUPPLYDEPOT).ready.exists:
            supplydepot = self.units(SUPPLYDEPOT).ready.random

            if self.can_afford(BARRACKS) and len(self.units(BARRACKS)) < self.MAX_BARRACKS:
                if self.can_afford(BARRACKS) and not self.already_pending(BARRACKS):
                    await self.build(BARRACKS, near=supplydepot)

            if self.units(BARRACKS).ready.exists and self.can_afford(FACTORY) and self.units(FACTORY).amount < self.MAX_FACTORIES:
                if self.can_afford(FACTORY) and not self.already_pending(FACTORY):
                    await self.build(FACTORY, near=supplydepot)

            if self.units(STARPORT).ready.exists and self.can_afford(STARPORT) and self.units(STARPORT).amount < self.MAX_STARPORTS:
                if self.can_afford(STARPORT) and not self.already_pending(STARPORT):
                    await self.build(STARPORT, near=supplydepot)

    async def buildings_attachments(self):
        if random.randint(0, 1) == 0:
            if self.units(FACTORY).ready:
                for factory in self.units(FACTORY).ready:
                    if self.can_afford(FACTORYTECHLAB) and factory.add_on_tag == 0 and self.can_place(FACTORYTECHLAB, factory.position) and self.units(FACTORY).amount < self.MAX_FACTORIES:
                        await self.do(factory.build(FACTORYTECHLAB, factory.add_on_land_position)) 
                    elif self.units(FACTORY).amount >= self.MAX_FACTORIES and factory.add_on_tag == 0 and self.can_afford(FACTORYREACTOR):
                        await self.do(factory.build(FACTORYREACTOR, factory.add_on_land_position))
        else:
            if self.units(BARRACKS).ready:
                for barrack in self.units(BARRACKS).ready:
                    if self.can_afford(BARRACKSTECHLAB) and barrack.add_on_tag == 0 and self.can_place(BARRACKSTECHLAB, barrack.position) and self.units(BARRACKS).amount < self.MAX_BARRACKS:
                        await self.do(barrack.build(BARRACKSTECHLAB, barrack.add_on_land_position)) 
                    elif self.units(BARRACKS).amount >= self.MAX_BARRACKS and barrack.add_on_tag == 0 and self.can_afford(BARRACKSREACTOR):
                        await self.do(barrack.build(BARRACKSREACTOR, barrack.add_on_land_position))

    async def train_offensive_force(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if not self.units(MARINE).amount > self.units(HELLION).amount * 3:
                if self.can_afford(MARINE) and self.supply_left > 0:
                    await self.do(barrack.train(MARINE))

        for factory in self.units(FACTORY).ready.noqueue:
            if self.can_afford(HELLION) and self.supply_left > 0:
                await self.do(factory.train(HELLION))

    def find_target(self, state):
            if len(self.known_enemy_units) > 0:
                return random.choice(self.known_enemy_units)
            elif len(self.known_enemy_structures) > 0:
                return random.choice(self.known_enemy_structures)
            else:
                return self.enemy_start_locations[0]

    async def attack(self):
        aggressive_units = {MARINE: [15, 5],
                            HELLION: [8, 3]}

        for UNIT in aggressive_units:
            if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.find_target(self.state)))

            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))

    async def build_defenses(self):
        if len(self.units(COMMANDCENTER)):
            commandcenter =  self.units(COMMANDCENTER)[random.randint(0, len(self.units(COMMANDCENTER)) - 1)]
            if self.can_afford(UnitTypeId.MISSILETURRET) and len(self.units(UnitTypeId.MISSILETURRET)) < self.MAX_MISSILETURRETS: 
                await self.build(UnitTypeId.PHOTONCANNON, near=commandcenter)

if __name__ == '__main__':
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Terran, TerranCustomBot()),
        Computer(Race.Protoss, Difficulty.Hard)
        ], realtime=False)