'''Binary heap class'''
class binary_heap:
    @classmethod
    def heappush(self, heap: list, item):
        """Pushes item onto heap"""
        heap.append(item)
        self._sift_up(heap, len(heap) - 1)

    @classmethod
    def heappop(self, heap: list):
        """Pops item from top of heap"""
        if len(heap) == 0:
            raise IndexError("Heap is empty")

        self._swap(heap, 0, len(heap) - 1)
        min_item = heap.pop()
        self._sift_down(heap, 0)
        return min_item

    @classmethod
    def _swap(self, heap, i, j):
        heap[i], heap[j] = heap[j], heap[i]

    @classmethod
    def _sift_up(self, heap, index):
        parent_index = (index - 1) // 2
        while index > 0 and heap[index] < heap[parent_index]:
            self._swap(heap, index, parent_index)
            index = parent_index
            parent_index = (index - 1) // 2

    @classmethod
    def _sift_down(self, heap, index):
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2

        while left_child_index < len(heap):
            smaller_child_index = left_child_index
            if right_child_index < len(heap) and heap[right_child_index] < heap[left_child_index]:
                smaller_child_index = right_child_index

            if heap[index] < heap[smaller_child_index]:
                break
            
            self._swap(heap, index, smaller_child_index)
            index = smaller_child_index
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2

