class Model:
    def eval(self, sample):
        """
        Args:
            sample (dict): Canonical Minecraft-HRL-Agent sample with keys such as
                id, biome, nearby_structures, y_level, task, and reasoning_path.

        Returns:
            list[str]: Predicted reasoning path in the canonical skill vocabulary.
        """
        raise NotImplementedError
