from heapq import heappush, heappop
import itertools
class PriorityQueue:
    def __init__(self):
        pq = []
        self.entry_finder = {}
        self.counter = itertools.count()
    
    def __len__(self):
        return len(self.pq)
    
    def add_task(self, priority, task):
        if task in self.entry_finder:
            self.update_priority(priority, task)
            return self
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def update(self, task, priority=0):
        '''Add a new task or update the priority of an existing task
        '''
        if task in self.entry_finder:
            self.remove_task(task)

        self.count += 1
        entry = [priority, self.count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')