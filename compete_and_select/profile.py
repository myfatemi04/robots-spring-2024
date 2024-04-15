import time

call_history = {}

def profile(tag):
    if tag not in call_history:
        call_history[tag] = []

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            call_history[tag].append((start_time, end_time))
            durations = [e - s for (s, e) in call_history[tag][-10:]]
            avg_duration = sum(durations)/len(durations)
            
            print(f"### Profile :: {tag} ###")
            print(f"time :: {avg_duration:.2f}s")
            print(f"hz :: {1/avg_duration:.2f}s")

            return result

        return wrapper

    return decorator
