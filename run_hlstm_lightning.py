from pytorch_lightning import Trainer

from hierarchical_model_lightning import HierarchicalModel

model = HierarchicalModel()

trainer = Trainer(gpus=8, num_nodes=1)
# other params like 'max_epochs' can be found https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
trainer.fit(model)
