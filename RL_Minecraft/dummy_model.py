class DummyModel:
    def eval(self, sample):
        task = sample["task"]
        structures = sample.get("nearby_structures", ["none"])

        if task == "obtain_iron_pickaxe":
            if "blacksmith" in structures:
                return ["go_to_village", "find_blacksmith", "loot_blacksmith_chest"]
            return [
                "harvest_wood",
                "craft_planks_and_sticks",
                "craft_crafting_table",
                "craft_wooden_pickaxe",
                "mine_stone",
                "craft_stone_pickaxe",
                "mine_iron_ore",
                "craft_furnace",
                "smelt_iron",
                "craft_iron_pickaxe",
            ]

        if task == "obtain_food":
            if "village" in structures:
                return ["go_to_village", "harvest_village_crops"]
            return ["search_for_animals", "kill_animals_for_meat", "cook_meat"]

        if task == "build_shelter":
            return [
                "harvest_wood",
                "craft_planks_and_sticks",
                "build_walls_and_roof",
                "craft_and_place_door",
                "place_torches",
            ]

        return []
