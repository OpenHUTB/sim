# Fix chumpy compatibility with Python 3.11+
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
    print("Patched inspect.getargspec")
