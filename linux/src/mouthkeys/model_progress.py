"""Machine-readable progress from the Hub's supported tqdm callback interface."""


def progress_tqdm(callback):
    from tqdm.auto import tqdm

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self.report()

        def display(self, *args, **kwargs):
            return None  # UI/CLI renders structured events, not terminal escape sequences.

        def report(self):
            if hasattr(self, "n"):
                callback(
                    {
                        "phase": "download",
                        "completed": self.n,
                        "total": self.total,
                        "unit": self.unit,
                        "description": self.desc,
                    }
                )

        def update(self, n=1):
            result = super().update(n)
            self.report()
            return result

    return Progress
