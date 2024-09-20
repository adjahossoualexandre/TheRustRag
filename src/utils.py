import os

class ManualExperiment():

    def __init__(self, file_name: str, dir: str, exp_metadata: dict = None, rm_existing_file: bool =True) -> None:
        self.file_name = file_name
        self.dir = dir
        self.exp_metadata = exp_metadata
        self.file_path = self.dir + "/" + self.file_name
        self.rm_existing_file = rm_existing_file

        if self.exp_metadata:
            if os.path.isfile(self.file_path) and self.rm_existing_file:
                os.remove(self.file_path)
            with open(self.file_path, mode="x") as f:
                for exp_m, value in self.exp_metadata.items():
                    f.writelines(f"{exp_m}: {value}")
                    f.writelines("\n")
                f.writelines("-"*24)
                f.writelines("\n")

    def track(self, output: dict) -> None:
        with open(self.file_path, mode="a") as f:
            for key, value in output.items():
                f.writelines(f"{key}: {value}")
                f.writelines("\n")
            f.writelines("-"*48 + "\n")
            f.writelines("-"*48 + "\n")
            f.writelines("\n")
        f.close()
