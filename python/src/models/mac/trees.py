"""
Tree-based models.
"""
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from .base import ClassicalModel

class DecisionTreeModel(ClassicalModel):
    def __init__(self, task='regression', **kwargs):
        super().__init__()
        if task == 'regression':
            self.model = DecisionTreeRegressor(**kwargs)
        else:
            self.model = DecisionTreeClassifier(**kwargs)

class RandomForestModel(ClassicalModel):
    def __init__(self, task='regression', **kwargs):
        super().__init__()
        if task == 'regression':
            self.model = RandomForestRegressor(**kwargs)
        else:
            self.model = RandomForestClassifier(**kwargs)

class GradientBoostingModel(ClassicalModel):
    def __init__(self, task='regression', **kwargs):
        super().__init__()
        if task == 'regression':
            self.model = GradientBoostingRegressor(**kwargs)
        else:
            self.model = GradientBoostingClassifier(**kwargs)
