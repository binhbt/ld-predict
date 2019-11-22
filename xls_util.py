import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def load_data(file_name, sheet):
	df = pd.read_excel(file_name, sheet_name=sheet)
	# print("Column headings:")
	# print(df.columns)
	data_list =[]
	for i in range(1,12):
		for j in df.index:
			# print(df['Th'+str(i)][j])
			data = str(df['Th'+str(i)][j])
			if data:
				data = data.replace('.0','')
				# print(data)
				if isInt(data):
					num = int(data)%100
					data_list.append(num)
					print(num)
		print('-------------------')
	return data_list

print(load_data('LOTO DATA.xlsx', 17))
