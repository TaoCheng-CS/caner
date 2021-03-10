domain_dict = {}
s_text = ''
t_text = ''

with open('dev.txt', 'r', encoding='utf-8') as f:
	span = ''
	do = ''
	for i, row in enumerate(f.readlines()):	
		if len(row) >= 2:	
			s_label = row.split()[0]
			t_label = row.split()[3]
				
		if len(row) < 2:
			s_text += '\n'
			t_text += '\n'
			continue
		s_text += s_label + ' '
		t_text += t_label + ' '
				
with open('dev_split.txt', 'w', encoding='utf-8') as f:
	f.write(s_text)
	
with open('dev_target.txt', 'w', encoding='utf-8') as f:
	f.write(t_text)
	