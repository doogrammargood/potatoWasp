def displayColor(color):
	new_color = list(color)
	print map(lambda c: 8*c+4, new_color)
	return map(lambda c: 8*c+4, new_color)[::-1]

print displayColor([8,6,21])