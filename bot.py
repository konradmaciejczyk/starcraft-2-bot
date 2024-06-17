import sc2 
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2.constants import COMMANDCENTER, SCV, SUPPLYDEPOT, REFINERY, \
                          BARRACKS, FACTORY, STARPORT, FACTORYTECHLAB, \
                          FACTORYREACTOR, BARRACKSREACTOR, BARRACKSTECHLAB, \
                          MARINE, HELLION
from sc2.ids.unit_typeid import UnitTypeId
import random
import cv2
import numpy as np
import time
import keras


class TerranCustomBot(sc2.BotAI):
    def __init__(self, use_model = False):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50
        self.MAX_BARRACKS = 5
        self.MAX_FACTORIES = 5
        self.MAX_STARPORTS = 5
        self.MAX_MISSILETURRETS = 10
        self.wait = 0
        self.train_data = []
        self.use_model = use_model
        self.flipped = None

        if self.use_model:
            self.model = keras.models.load_model("D:\Studia II Stopień\Starcraft II project\Projekt\BasicCNN-30-epochs-0.0001-LR-4.2")

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

        await self.update_heatmap()

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

    # async def attack(self):
    #     aggressive_units = {MARINE: [15, 5],
    #                         HELLION: [8, 3]}

    #     for UNIT in aggressive_units:
    #         if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
    #             for s in self.units(UNIT).idle:
    #                 await self.do(s.attack(self.find_target(self.state)))

    #         elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
    #             if len(self.known_enemy_units) > 0:
    #                 for s in self.units(UNIT).idle:
    #                     await self.do(s.attack(random.choice(self.known_enemy_units)))

    async def attack(self):
        if len(self.units(MARINE).idle) > 0 or len(self.units(HELLION).idle):

            target = False
            if self.iteration > self.wait:
                if self.use_model:
                    prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                    choice = np.argmax(prediction[0])
                else:
                    choice = random.randrange(0, 4)


                if choice == 0:
                    # no attack
                    wait = random.randrange(20,165)
                    self.wait = self.iteration + wait

                elif choice == 1:
                    #attack_unit_closest_commandcenter
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(COMMANDCENTER)))

                elif choice == 2:
                    #attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    #Attack Enemy Start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(MARINE).idle:
                        await self.do(vr.attack(target))

                    for vr in self.units(HELLION).idle:
                        await self.do(vr.attack(target))

                y = np.zeros(4)
                y[choice] = 1
                self.train_data.append([y,self.flipped])

    async def build_defenses(self):
        if len(self.units(COMMANDCENTER)):
            commandcenter =  self.units(COMMANDCENTER)[random.randint(0, len(self.units(COMMANDCENTER)) - 1)]
            if self.can_afford(UnitTypeId.MISSILETURRET) and len(self.units(UnitTypeId.MISSILETURRET)) < self.MAX_MISSILETURRETS: 
                await self.build(UnitTypeId.PHOTONCANNON, near=commandcenter)

    async def update_heatmap(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        draw_dict = {
                     COMMANDCENTER: [15, (0, 255, 0)],
                     SUPPLYDEPOT: [3, (20, 235, 0)],
                     SCV: [1, (55, 200, 0)],
                     REFINERY: [2, (55, 200, 0)],
                     BARRACKS: [3, (200, 100, 0)],
                     FACTORY: [3, (150, 150, 0)],
                     STARPORT: [5, (255, 0, 0)]
                    }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ["nexus", "supplydepot", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe", "scv", "drone"]    

                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(MARINE)) / (self.supply_cap-self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        cv2.imshow('', resized)
        cv2.waitKey(1)

    def on_end(self, game_result):
        print(game_result)

        if game_result == Result.Victory:
            # np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))   
            pass     

if __name__ == '__main__':
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Terran, TerranCustomBot(use_model=True)),
        Computer(Race.Protoss, Difficulty.Easy)
        ], realtime=True)