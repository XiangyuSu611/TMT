"""
##@package datastructures
##@brief This module contains specialised containers. 
##
#These are used to implement algorithms cleanly.
"""
from random import randint

##@brief A class that works just like a queue or a stack, except
## that a randomly selected element is returned. 
##
# This class is useful for implementing algorithms that gather 
# elements, and need to process them randomly. Something along the 
# lines of:
# 
# @code
# while not rqueue.empty():
#   #generates 3 new elements to process
#   for i in range(3): 
#     rqueue.push(process(rqueue.pop())) 
# @endcode
class RandomQueue:
	## Constructs a new empty RandomQueue
	def __init__(self):		
		## The internal list to store objects.
		self.array = []

	##Returns True if this RandomQueue is empty.	
	def empty(self):
		return len(self.array) <= 0
	
	## Push a new element into the RandomQueue.
	def push(self, x):
		self.array.append(x)
	
	## @brief Pops a randomly selected element from the queue. 
	##
	# All elements can be selected equiprobably
	def pop(self):
		n = len(self.array)
		
		if n <= 0:
			raise IndexError('Cannot pop from emty container!')
		elif n == 1:
			return self.array.pop()
		else:
			i = randint(0, n - 1)
			j = n - 1
			self.array[i], self.array[j] = 	self.array[j], self.array[i]

		return self.array.pop()