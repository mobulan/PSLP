class Log:
	def __init__(self, fname=None, write=True):
		super(Log, self).__init__()
		self.write = write
		self.fname = fname

		# Create the file
		if self.write:
			with open(self.fname, 'w') as f:
				pass

	def info(self, *info, end='\n'):
		print(*info, flush=True, end=end)
		if self.write:
			with open(self.fname, 'a+') as f:
				print(*info, file=f, flush=True, end=end)


class PMarkdownTable:
	def __init__(self, log, titles, rank=0):
		if rank not in [-1, 0]: return
		super().__init__()
		title_line = '| '
		align_line = '| '
		for i in range(len(titles)):
			title_line = title_line + titles[i] + ' |'
			align_line = align_line + '--- |'
		log.markdown(title_line)
		log.markdown(align_line)

	def add(self, log, values, rank=0):
		if rank not in [-1, 0]: return
		value_line = '| '
		for i in range(len(values)):
			value_line = value_line + str(values[i]) + '|'
		log.markdown(value_line)