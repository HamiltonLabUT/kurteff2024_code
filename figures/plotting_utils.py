




def rgb_to_hex(r, g, b, percent_value = False):
	'''
	Converts a 1-256 RGB value into hexadecimal format.
	* percent_value: bool for whether you are scaling RGB from 0-1 (True) or 1-256 (False).
	'''
	if percent_value:
		r = int(r * 255)
		g = int(g * 255)
		b = int(b * 255)
	return '#{:02x}{:02x}{:02x}'.format(r, g, b)