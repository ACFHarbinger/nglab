import cProfile
import pstats
import io
from functools import wraps
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def profile(output_dir: str = "./profiles"):
    """
    Decorator to profile function execution using cProfile.
    Writes results to a .prof file in the specified directory.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            prof = cProfile.Profile()
            prof.enable()
            result = func(*args, **kwargs)
            prof.disable()
            
            # Save raw stats
            stats_file = out_path / f"{func.__name__}.prof"
            prof.dump_stats(str(stats_file))
            
            # Log summary
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats(20)
            logger.info(f"Performance profile for {func.__name__}:\n{s.getvalue()}")
            
            return result
        return wrapper
    return decorator
