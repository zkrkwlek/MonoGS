import queue

class PeekableQueue(queue.Queue):
    def peek(self):
        with self.mutex:
            if self._qsize() > 0:
                return self.queue[0]
            else:
                raise queue.Empty

    def __len__(self):
        return self.qsize()