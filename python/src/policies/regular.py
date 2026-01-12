from .base import Policy

class RegularPolicy(Policy):
    """
    Regular policy that executes an action every N steps.
    Ported conceptually from logic/src/policies/regular.py (which was bin collection based).
    Here adapted for trading: e.g. Rebalance every N days.
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.period = self.cfg.get('period', 30) # collection every (lvl+1) days
        self.current_step = 0

    def act(self, observation):
        self.current_step += 1
        
        if self.current_step % self.period == 0:
            return 1 # Action (e.g. Buy/Rebalance)
        
        return 0 # Do nothing
    
    def reset(self):
        self.current_step = 0
